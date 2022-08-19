/// Module containing architecture-specific utility code.

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) mod neon;
