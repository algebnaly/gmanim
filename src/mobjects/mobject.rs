use nalgebra::Matrix4;

use crate::GMFloat;

use super::submission::RenderVisitor;

pub trait Mobject: Send + Sync + 'static {
    fn default_name(&self) -> &'static str;

    fn submit_to_renderer(
        &self,
        _visitor: &mut dyn RenderVisitor,
        _world_transform: Matrix4<GMFloat>,
    ) {
    }
}
