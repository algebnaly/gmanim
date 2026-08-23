#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameRange {
    pub start: u32,
    pub end: u32,
}

impl FrameRange {
    pub fn duration(self) -> u32 {
        self.end - self.start
    }

    pub(super) fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}
