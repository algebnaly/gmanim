use std::fs;
use std::process::{Command, Stdio};

pub const TYPST_HEADER: &str = r#"#set page(
    width: auto,
    height: auto,
    margin: (x: 0cm, y: 0cm)
)"#;
pub struct Formula {
    formula_text: String,
}

impl Formula {
    pub fn new(formula_text: &str) -> Self {
        Self {
            formula_text: formula_text.to_owned(),
        }
    }

    fn write_to_typst(&self, output_typst_file_path: &str) {
        fs::write(
            output_typst_file_path,
            format!("{TYPST_HEADER}\n$ {} $", self.formula_text),
        )
        .unwrap();
    }

    pub fn to_mobject(&self) -> crate::mobjects::group::MobjectGroup {
        let typst_path = "formula_temp.typst";
        let svg_path = "formula_temp.svg";
        self.write_to_typst(typst_path);
        compile_to_svg(typst_path, svg_path);

        let group = crate::mobjects::svg_shape::open_svg_file(svg_path);

        // Clean up temp files
        let _ = fs::remove_file(typst_path);
        let _ = fs::remove_file(svg_path);

        group
    }
}

pub fn compile_to_svg(typst_file_path: &str, svg_file_path: &str) {
    let mut c = Command::new("typst")
        .args(["compile", "-f", "svg", typst_file_path, svg_file_path])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("can't compile to svg");
    c.wait().expect("typst execution failed");
}

#[test]
fn test_formula_to_mobject() {
    let f = Formula::new("pi");
    let mut mobj = f.to_mobject();

    // The SVGShape parser should have populated the elements,
    // but update_mesh() must be called to populate the mesh.
    // In our open_svg_file, update_mesh is already called!
    use crate::mobjects::Mobject;

    // A pi symbol has paths, so we should have some vertices.
    let mut vertex_count = 0;
    for child in mobj.mobjects.iter() {
        if let Some(mesh) = child.as_mesh_2d() {
            vertex_count += mesh.vertices.len();
        }
    }

    assert!(
        vertex_count > 0,
        "Formula parsing failed to generate vertices"
    );
}
