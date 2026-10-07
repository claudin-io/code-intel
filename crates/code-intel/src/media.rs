//! Which workspace files are media the index can describe: images and audio.
//!
//! This module only *recognizes* media (by extension, with the same ignore
//! rules as the code scan) and holds the knobs. It has no dependencies of its
//! own and compiles in every build, so a workspace's images and sounds are
//! always findable by file name. Turning their content into vectors is
//! `media_prep` + `gemma2`, which exist only in the ONNX Runtime build.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Image,
    Audio,
}

impl MediaKind {
    pub const ALL: [MediaKind; 2] = [MediaKind::Image, MediaKind::Audio];

    /// The value stored as `symbols.kind` and `files.language` for a media
    /// file — one vocabulary for the index, the search results and the tools.
    pub fn as_str(self) -> &'static str {
        match self {
            MediaKind::Image => "image",
            MediaKind::Audio => "audio",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "image" => Some(MediaKind::Image),
            "audio" => Some(MediaKind::Audio),
            _ => None,
        }
    }
}

/// SQL list of the media symbol kinds, for `kind IN (...)` filters.
pub const MEDIA_KINDS_SQL: &str = "'image','audio'";

/// Raster formats the pure-Rust decoders read. SVG is deliberately absent: it
/// is XML, already indexed as code by tree-sitter.
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "gif", "bmp"];
/// AAC (`.aac`, `.m4a`) is left out on purpose. Files in a repository are
/// untrusted input to the decoder, and under fuzzing the AAC/MP4 readers were
/// the ones that did not hold: a corrupted `.m4a` asked for a 64 GB
/// allocation, which aborts the process — nothing can catch that. The formats
/// listed here took thousands of corrupted inputs with nothing worse than an
/// error.
const AUDIO_EXTS: &[&str] = &["wav", "mp3", "flac", "ogg", "oga"];

pub fn media_kind(path: &str) -> Option<MediaKind> {
    let ext = Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
    if IMAGE_EXTS.contains(&ext.as_str()) {
        Some(MediaKind::Image)
    } else if AUDIO_EXTS.contains(&ext.as_str()) {
        Some(MediaKind::Audio)
    } else {
        None
    }
}

/// Files above these sizes are still listed (findable by name) but never
/// decoded: a 200 MB texture atlas or an hour-long recording is not what the
/// index is for, and decoding it costs unbounded memory.
pub const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;
pub const MAX_AUDIO_BYTES: u64 = 64 * 1024 * 1024;

pub fn max_bytes(kind: MediaKind) -> u64 {
    match kind {
        MediaKind::Image => MAX_IMAGE_BYTES,
        MediaKind::Audio => MAX_AUDIO_BYTES,
    }
}

/// Which media encoders a workspace needs — or, on an embedder, which it has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaNeeds {
    pub images: bool,
    pub audio: bool,
}

impl MediaNeeds {
    pub const NONE: MediaNeeds = MediaNeeds {
        images: false,
        audio: false,
    };

    pub fn has(self, kind: MediaKind) -> bool {
        match kind {
            MediaKind::Image => self.images,
            MediaKind::Audio => self.audio,
        }
    }

    pub fn set(&mut self, kind: MediaKind) {
        match kind {
            MediaKind::Image => self.images = true,
            MediaKind::Audio => self.audio = true,
        }
    }

    pub fn any(self) -> bool {
        self.images || self.audio
    }

    pub fn union(self, other: MediaNeeds) -> MediaNeeds {
        MediaNeeds {
            images: self.images || other.images,
            audio: self.audio || other.audio,
        }
    }
}

/// `CODE_INTEL_MEDIA=0` turns media indexing off entirely: no media rows, no
/// encoder downloads.
pub fn media_enabled() -> bool {
    std::env::var("CODE_INTEL_MEDIA")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// How many media files per workspace get a content vector. Encoding an image
/// is seconds of CPU, not milliseconds like a code chunk, so a repository of
/// ten thousand screenshots must not turn the first index into a day-long
/// job. Files past the cap stay findable by name. `CODE_INTEL_MEDIA_MAX`
/// overrides.
pub const DEFAULT_MAX_EMBEDDED_MEDIA: usize = 200;

pub fn max_embedded_media() -> usize {
    std::env::var("CODE_INTEL_MEDIA_MAX")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_MAX_EMBEDDED_MEDIA)
}

fn walker(root: &str) -> ignore::Walk {
    // Same rules as `indexer::scan_workspace`: what the code scan skips
    // (.gitignore, hidden directories) is skipped here too.
    ignore::WalkBuilder::new(root)
        .git_ignore(true)
        .git_global(true)
        .hidden(true)
        .build()
}

/// Every media file under `root`, sorted by path so the cap above always
/// selects the same files.
pub fn media_files(root: &str) -> Vec<(String, MediaKind)> {
    if !media_enabled() {
        return Vec::new();
    }
    let mut out: Vec<(String, MediaKind)> = walker(root)
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|e| {
            let p = e.path().to_string_lossy().to_string();
            media_kind(&p).map(|k| (p, k))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Whether the workspace holds any image and any audio at all. This is what
/// decides which encoders are worth downloading; it stops walking as soon as
/// both answers are yes.
pub fn detect_media_needs(root: &str) -> MediaNeeds {
    let mut needs = MediaNeeds::NONE;
    if !media_enabled() {
        return needs;
    }
    for entry in walker(root).filter_map(|e| e.ok()) {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        if let Some(kind) = media_kind(&entry.path().to_string_lossy()) {
            needs.set(kind);
            if needs.images && needs.audio {
                break;
            }
        }
    }
    needs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_come_from_the_extension_case_insensitively() {
        assert_eq!(media_kind("assets/Logo.PNG"), Some(MediaKind::Image));
        assert_eq!(media_kind("a/b/photo.jpeg"), Some(MediaKind::Image));
        assert_eq!(media_kind("sfx/click.wav"), Some(MediaKind::Audio));
        assert_eq!(media_kind("music/theme.FLAC"), Some(MediaKind::Audio));
        // SVG is code (XML), video is out of scope, AAC is not decoded (see
        // AUDIO_EXTS), and the rest is not media.
        for other in [
            "voice.m4a",
            "voice.aac",
            "icon.svg",
            "demo.mp4",
            "main.rs",
            "README.md",
            "noext",
            "archive.zip",
        ] {
            assert_eq!(media_kind(other), None, "{other}");
        }
    }

    #[test]
    fn kind_names_round_trip() {
        for kind in MediaKind::ALL {
            assert_eq!(MediaKind::parse(kind.as_str()), Some(kind));
            assert!(MEDIA_KINDS_SQL.contains(&format!("'{}'", kind.as_str())));
        }
        assert_eq!(MediaKind::parse("function_item"), None);
    }

    /// The download decision: a text-only project needs no encoder, a project
    /// with images needs only the vision one, and ignored files never count.
    #[test]
    fn needs_reflect_what_the_workspace_actually_contains() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().to_string();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        assert_eq!(detect_media_needs(&root), MediaNeeds::NONE);
        assert!(media_files(&root).is_empty());

        std::fs::create_dir_all(dir.path().join("assets")).unwrap();
        std::fs::write(dir.path().join("assets/logo.png"), b"not really a png").unwrap();
        assert_eq!(
            detect_media_needs(&root),
            MediaNeeds {
                images: true,
                audio: false
            }
        );

        // Ignored and hidden media must not trigger a download.
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "build/\n").unwrap();
        std::fs::create_dir_all(dir.path().join("build")).unwrap();
        std::fs::write(dir.path().join("build/beep.wav"), b"x").unwrap();
        std::fs::create_dir_all(dir.path().join(".cache")).unwrap();
        std::fs::write(dir.path().join(".cache/beep.mp3"), b"x").unwrap();
        assert!(!detect_media_needs(&root).audio);

        std::fs::write(dir.path().join("assets/click.ogg"), b"x").unwrap();
        let needs = detect_media_needs(&root);
        assert!(needs.images && needs.audio);
        let files = media_files(&root);
        assert_eq!(files.len(), 2);
        assert!(files[0].0 < files[1].0, "sorted by path");
    }
}
