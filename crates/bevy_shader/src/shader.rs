use super::ShaderDefVal;
use alloc::borrow::Cow;
use bevy_asset::{io::Reader, Asset, AssetLoader, AssetPath, Handle, LoadContext};
use bevy_reflect::TypePath;
use bevy_utils::define_atomic_id;
use thiserror::Error;

/// The last import segment may name a module or an item, so each import is
/// emitted in both forms.
#[cfg(feature = "shader_format_wesl")]
fn scan_wesl_imports(
    source: &str,
    self_module_path: &wesl::syntax::ModulePath,
) -> Vec<ShaderImport> {
    use wesl::syntax::{ImportContent, ModulePath, PathOrigin};

    fn leaves(content: &ImportContent, path: ModulePath, out: &mut Vec<ModulePath>) {
        match content {
            ImportContent::Item(item) => {
                let mut full = path.clone();
                full.push(&item.ident.to_string());
                out.push(path);
                out.push(full);
            }
            ImportContent::Collection(collection) => {
                for import in collection {
                    let path = path.clone().join(import.path.iter().cloned());
                    leaves(&import.content, path, out);
                }
            }
        }
    }

    let Ok(translation_unit) = source.parse::<wesl::syntax::TranslationUnit>() else {
        return Vec::new();
    };

    let mut paths = Vec::new();
    for statement in &translation_unit.imports {
        match &statement.path {
            Some(import_path) => {
                let path = self_module_path.join_path(import_path);
                leaves(&statement.content, path, &mut paths);
            }
            None => {
                if let ImportContent::Collection(collection) = &statement.content {
                    for import in collection {
                        let mut components = import.path.iter().cloned();
                        if let Some(package) = components.next() {
                            let path =
                                ModulePath::new(PathOrigin::Package(package), components.collect());
                            leaves(&import.content, path, &mut paths);
                        }
                    }
                }
            }
        }
    }

    let mut imports = Vec::new();
    for path in &paths {
        let path = crate::shader_cache::canonicalize_module_path(path);
        let import = match &path.origin {
            PathOrigin::Absolute => {
                ShaderImport::AssetPath(format!("/{}", path.components.join("/")))
            }
            PathOrigin::Package(package) => ShaderImport::Custom(
                core::iter::once(package.as_str())
                    .chain(path.components.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join("::"),
            ),
            PathOrigin::Relative(_) => continue,
        };
        if !imports.contains(&import) {
            imports.push(import);
        }
    }
    imports
}

define_atomic_id!(ShaderId);

/// Describes whether or not to perform runtime checks on shaders.
/// Runtime checks can be enabled for safety at the cost of speed.
/// By default no runtime checks will be performed.
///
/// # Panics
/// Because no runtime checks are performed for spirv,
/// enabling `ValidateShader` for spirv will cause a panic
#[derive(Clone, Debug, Default)]
pub enum ValidateShader {
    #[default]
    /// No runtime checks for soundness (e.g. bound checking) are performed.
    ///
    /// This is suitable for trusted shaders, written by your program or dependencies you trust.
    Disabled,
    /// Enable's runtime checks for soundness (e.g. bound checking).
    ///
    /// While this can have a meaningful impact on performance,
    /// this setting should *always* be enabled when loading untrusted shaders.
    /// This might occur if you are creating a shader playground, running user-generated shaders
    /// (as in `VRChat`), or writing a web browser in Bevy.
    Enabled,
}

/// An "unprocessed" shader. It can contain preprocessor directives and imports.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct Shader {
    /// The asset path of the shader.
    pub path: String,
    /// The raw source code of the shader.
    pub source: Source,
    /// The path from which this shader can be imported by other shaders.
    pub import_path: ShaderImport,
    /// The import paths this shader depends on.
    pub imports: Vec<ShaderImport>,
    /// Extra imports not specified in the source string.
    pub additional_imports: Vec<naga_oil::compose::ImportDefinition>,
    /// Any shader defs that should be included when this module is used.
    pub shader_defs: Vec<ShaderDefVal>,
    /// Strong handles to this shader's dependencies, to prevent them
    /// from being immediately dropped if this shader is the only user.
    pub file_dependencies: Vec<Handle<Shader>>,
    /// Enable or disable runtime shader validation, trading safety against speed.
    ///
    /// Please read the [`ValidateShader`] docs for a discussion of the tradeoffs involved.
    pub validate_shader: ValidateShader,
}

impl Shader {
    fn preprocess(source: &str, path: &str) -> (ShaderImport, Vec<ShaderImport>) {
        let (import_path, imports, _) = naga_oil::compose::get_preprocessor_data(source);

        let import_path = import_path
            .map(ShaderImport::Custom)
            .unwrap_or_else(|| ShaderImport::AssetPath(path.to_owned()));

        let imports = imports
            .into_iter()
            .map(|import| {
                if import.import.starts_with('\"') {
                    let import = import
                        .import
                        .chars()
                        .skip(1)
                        .take_while(|c| *c != '\"')
                        .collect();
                    ShaderImport::AssetPath(import)
                } else {
                    ShaderImport::Custom(import.import)
                }
            })
            .collect();

        (import_path, imports)
    }

    /// Creates a new WGSL shader.
    pub fn from_wgsl(source: impl Into<Cow<'static, str>>, path: impl Into<String>) -> Shader {
        let source = source.into();
        let path = path.into();
        let (import_path, imports) = Shader::preprocess(&source, &path);
        Shader {
            path,
            imports,
            import_path,
            source: Source::Wgsl(source),
            additional_imports: Default::default(),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }

    /// Creates a new WGSL shader with some given shader defs.
    pub fn from_wgsl_with_defs(
        source: impl Into<Cow<'static, str>>,
        path: impl Into<String>,
        shader_defs: Vec<ShaderDefVal>,
    ) -> Shader {
        Self {
            shader_defs,
            ..Self::from_wgsl(source, path)
        }
    }

    /// Creates a new GLSL shader.
    pub fn from_glsl(
        source: impl Into<Cow<'static, str>>,
        stage: naga::ShaderStage,
        path: impl Into<String>,
    ) -> Shader {
        let source = source.into();
        let path = path.into();
        let (import_path, imports) = Shader::preprocess(&source, &path);
        Shader {
            path,
            imports,
            import_path,
            source: Source::Glsl(source, stage),
            additional_imports: Default::default(),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }

    /// Creates a new SPIR-V shader.
    pub fn from_spirv(source: impl Into<Cow<'static, [u8]>>, path: impl Into<String>) -> Shader {
        let path = path.into();
        Shader {
            path: path.clone(),
            imports: Vec::new(),
            import_path: ShaderImport::AssetPath(path),
            source: Source::SpirV(source.into()),
            additional_imports: Default::default(),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }

    /// Creates a new Wesl shader.
    #[cfg(feature = "shader_format_wesl")]
    pub fn from_wesl(source: impl Into<Cow<'static, str>>, path: impl Into<String>) -> Shader {
        Self::from_wesl_with_import_path(source, path, None::<String>)
    }

    /// Creates a new Wesl shader, registered under the logical `import_path` if
    /// provided, otherwise under its asset path.
    #[cfg(feature = "shader_format_wesl")]
    pub fn from_wesl_with_import_path(
        source: impl Into<Cow<'static, str>>,
        path: impl Into<String>,
        import_path: Option<impl Into<String>>,
    ) -> Shader {
        let source = source.into();
        let path = path.into();

        let import_path = match import_path {
            Some(import_path) => ShaderImport::Custom(import_path.into()),
            None => {
                // Create the shader import path - always starting with "/"
                let shader_path = std::path::Path::new("/").join(&path);

                // Convert to a string with forward slashes and without extension
                let import_path_str = shader_path
                    .with_extension("")
                    .to_string_lossy()
                    .replace('\\', "/");

                ShaderImport::AssetPath(import_path_str.to_string())
            }
        };

        let imports = crate::shader_cache::wesl_module_path(&import_path)
            .map(|module_path| scan_wesl_imports(&source, &module_path))
            .unwrap_or_default();

        Shader {
            path,
            imports,
            import_path,
            source: Source::Wesl(source),
            additional_imports: Default::default(),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }
}

impl<'a> From<&'a Shader> for naga_oil::compose::ComposableModuleDescriptor<'a> {
    fn from(shader: &'a Shader) -> Self {
        let shader_defs = shader
            .shader_defs
            .iter()
            .map(|def| match def {
                ShaderDefVal::Bool(name, b) => (
                    name.to_string(),
                    naga_oil::compose::ShaderDefValue::Bool(*b),
                ),
                ShaderDefVal::Int(name, i) => {
                    (name.to_string(), naga_oil::compose::ShaderDefValue::Int(*i))
                }
                ShaderDefVal::UInt(name, i) => (
                    name.to_string(),
                    naga_oil::compose::ShaderDefValue::UInt(*i),
                ),
            })
            .collect();

        // It is beyond me why this doesn't just use `shader.import_path.module_name()`.
        let as_name = match &shader.import_path {
            ShaderImport::AssetPath(asset_path) => Some(format!("\"{asset_path}\"")),
            ShaderImport::Custom(_) => None,
        };

        naga_oil::compose::ComposableModuleDescriptor {
            source: shader.source.as_str(),
            file_path: &shader.path,
            language: (&shader.source).into(),
            additional_imports: &shader.additional_imports,
            shader_defs,
            as_name,
        }
    }
}

impl<'a> From<&'a Shader> for naga_oil::compose::NagaModuleDescriptor<'a> {
    fn from(shader: &'a Shader) -> Self {
        naga_oil::compose::NagaModuleDescriptor {
            source: shader.source.as_str(),
            file_path: &shader.path,
            shader_type: (&shader.source).into(),
            ..Default::default()
        }
    }
}

/// Raw shader source code.
#[expect(missing_docs, reason = "The variants are self-explanatory.")]
#[derive(Debug, Clone)]
pub enum Source {
    Wgsl(Cow<'static, str>),
    Wesl(Cow<'static, str>),
    Glsl(Cow<'static, str>, naga::ShaderStage),
    SpirV(Cow<'static, [u8]>),
    // TODO: consider the following
    // PrecompiledSpirVMacros(HashMap<HashSet<String>, Vec<u32>>)
    // NagaModule(Module) ... Module impls Serialize/Deserialize
}

impl Source {
    /// The underlying source code string, unless it is SPIR-V.
    pub fn as_str(&self) -> &str {
        match self {
            Source::Wgsl(s) | Source::Wesl(s) | Source::Glsl(s, _) => s,
            Source::SpirV(_) => panic!("spirv not yet implemented"),
        }
    }
}

impl From<&Source> for naga_oil::compose::ShaderLanguage {
    fn from(value: &Source) -> Self {
        match value {
            Source::Wgsl(_) => naga_oil::compose::ShaderLanguage::Wgsl,
            #[cfg(any(feature = "shader_format_glsl", target_arch = "wasm32"))]
            Source::Glsl(_, _) => naga_oil::compose::ShaderLanguage::Glsl,
            #[cfg(all(not(feature = "shader_format_glsl"), not(target_arch = "wasm32")))]
            Source::Glsl(_, _) => panic!(
                "GLSL is not supported in this configuration; use the feature `shader_format_glsl`"
            ),
            Source::SpirV(_) => panic!("spirv not yet implemented"),
            Source::Wesl(_) => panic!("wesl not yet implemented"),
        }
    }
}

impl From<&Source> for naga_oil::compose::ShaderType {
    fn from(value: &Source) -> Self {
        match value {
            Source::Wgsl(_) => naga_oil::compose::ShaderType::Wgsl,
            #[cfg(any(feature = "shader_format_glsl", target_arch = "wasm32"))]
            Source::Glsl(_, shader_stage) => match shader_stage {
                naga::ShaderStage::Vertex => naga_oil::compose::ShaderType::GlslVertex,
                naga::ShaderStage::Fragment => naga_oil::compose::ShaderType::GlslFragment,
                naga::ShaderStage::Compute => panic!("glsl compute not yet implemented"),
                naga::ShaderStage::Task => panic!("task shaders not yet implemented"),
                naga::ShaderStage::Mesh => panic!("mesh shaders not yet implemented"),
                naga::ShaderStage::RayGeneration => {
                    panic!("ray generation shader not yet implemented")
                }
                naga::ShaderStage::Miss => panic!("miss shader not yet implemented"),
                naga::ShaderStage::AnyHit => panic!("any hit shader not yet implemented"),
                naga::ShaderStage::ClosestHit => panic!("closest hit shader not yet implemented"),
            },
            #[cfg(all(not(feature = "shader_format_glsl"), not(target_arch = "wasm32")))]
            Source::Glsl(_, _) => panic!(
                "GLSL is not supported in this configuration; use the feature `shader_format_glsl`"
            ),
            Source::SpirV(_) => panic!("spirv not yet implemented"),
            Source::Wesl(_) => panic!("wesl not yet implemented"),
        }
    }
}

/// The [`AssetLoader`] responsible for loading unprocessed shader assets.
#[derive(Default, TypePath)]
pub struct ShaderLoader;

/// An error encountered while loading a shader's source.
#[non_exhaustive]
#[derive(Debug, Error)]
#[expect(missing_docs, reason = "The variants are self-explanatory.")]
pub enum ShaderLoaderError {
    #[error("Could not load shader: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse shader: {0}")]
    Parse(#[from] alloc::string::FromUtf8Error),
}

/// Settings for loading shaders.
#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
pub struct ShaderSettings {
    /// The `#define`s specified for this shader.
    pub shader_defs: Vec<ShaderDefVal>,
    /// The logical import path to register this WESL shader under.
    #[serde(default)]
    pub import_path: Option<String>,
}

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ShaderSettings;
    type Error = ShaderLoaderError;
    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Shader, Self::Error> {
        let ext = load_context
            .path()
            .path()
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let ext = ext.as_str();
        let path = load_context.path().to_string();
        // On windows, the path will inconsistently use \ or /.
        // TODO: remove this once AssetPath forces cross-platform "slash" consistency. See #10511
        let path = path.replace(std::path::MAIN_SEPARATOR, "/");
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        if ext != "wgsl" && ext != "wesl" && !settings.shader_defs.is_empty() {
            tracing::warn!(
                "Tried to load a non-wgsl shader with shader defs, this isn't supported: \
                    The shader defs will be ignored."
            );
        }
        let mut shader = match ext {
            "spv" => Shader::from_spirv(bytes, load_context.path().path().to_string_lossy()),
            "wgsl" => Shader::from_wgsl_with_defs(
                String::from_utf8(bytes)?,
                path,
                settings.shader_defs.clone(),
            ),
            "vert" => Shader::from_glsl(String::from_utf8(bytes)?, naga::ShaderStage::Vertex, path),
            "frag" => {
                Shader::from_glsl(String::from_utf8(bytes)?, naga::ShaderStage::Fragment, path)
            }
            "comp" => {
                Shader::from_glsl(String::from_utf8(bytes)?, naga::ShaderStage::Compute, path)
            }
            #[cfg(feature = "shader_format_wesl")]
            "wesl" => {
                let mut shader = Shader::from_wesl_with_import_path(
                    String::from_utf8(bytes)?,
                    path,
                    settings.import_path.as_deref(),
                );
                shader.shader_defs = settings.shader_defs.clone();
                shader
            }
            _ => panic!("unhandled extension: {ext}"),
        };

        // collect and store file dependencies
        match ext {
            #[cfg(feature = "shader_format_wesl")]
            "wesl" => {
                let candidates: Vec<String> = shader
                    .imports
                    .iter()
                    .filter_map(|import| match import {
                        ShaderImport::AssetPath(asset_path) => {
                            Some(format!("{}.{ext}", asset_path.trim_start_matches('/')))
                        }
                        ShaderImport::Custom(_) => None,
                    })
                    .collect();
                for file_path in candidates {
                    if load_context
                        .read_asset_bytes(AssetPath::from(file_path.clone()))
                        .await
                        .is_ok()
                    {
                        shader
                            .file_dependencies
                            .push(load_context.load(AssetPath::from(file_path)));
                    }
                }
            }
            _ => {
                for import in &shader.imports {
                    if let ShaderImport::AssetPath(asset_path) = import {
                        shader.file_dependencies.push(load_context.load(asset_path));
                    }
                }
            }
        }
        Ok(shader)
    }

    fn extensions(&self) -> &[&str] {
        &["spv", "wgsl", "vert", "frag", "comp", "wesl"]
    }
}

/// A shader import, described as either an asset path or an import path.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ShaderImport {
    /// An asset path to a shader.
    AssetPath(String),
    /// An import path from which a shader may be imported.
    Custom(String),
}

impl ShaderImport {
    /// A name for a shader import.
    pub fn module_name(&self) -> Cow<'_, String> {
        match self {
            ShaderImport::AssetPath(s) => Cow::Owned(format!("\"{s}\"")),
            ShaderImport::Custom(s) => Cow::Borrowed(s),
        }
    }
}

/// A reference to a shader asset.
#[derive(Default)]
pub enum ShaderRef {
    /// Use the "default" shader for the current context.
    #[default]
    Default,
    /// A handle to a shader stored in the [`Assets<Shader>`](bevy_asset::Assets) resource.
    Handle(Handle<Shader>),
    /// An asset path leading to a shader.
    Path(AssetPath<'static>),
}

impl From<Handle<Shader>> for ShaderRef {
    fn from(handle: Handle<Shader>) -> Self {
        Self::Handle(handle)
    }
}

impl From<AssetPath<'static>> for ShaderRef {
    fn from(path: AssetPath<'static>) -> Self {
        Self::Path(path)
    }
}

impl From<&'static str> for ShaderRef {
    fn from(path: &'static str) -> Self {
        Self::Path(AssetPath::from(path))
    }
}
