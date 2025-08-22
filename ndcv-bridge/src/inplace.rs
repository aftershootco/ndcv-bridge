use opencv::core::Mat;
use opencv::prelude::*;
use opencv::Result;

#[inline(always)]
pub(crate) unsafe fn op_inplace<T>(
    m: &mut Mat,
    f: impl FnOnce(&Mat, &mut Mat) -> Result<T>,
) -> Result<T> {
    let mut m_alias = Mat::from_raw(m.as_raw_mut());
    let out = f(m, &mut m_alias);
    let _ = m_alias.into_raw();
    out
}
