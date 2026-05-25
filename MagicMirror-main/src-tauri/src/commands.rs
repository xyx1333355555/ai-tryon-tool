use std::path::Path;
use tauri::AppHandle;

use crate::utils::{download_file, unzip_file};

#[tauri::command]
pub fn file_exists(path: String) -> bool {
    Path::new(&path).exists()
}

#[tauri::command]
pub async fn download_and_unzip(
    app: AppHandle,
    url: String,
    target_dir: String,
) -> Result<(), String> {
    let temp_dir = std::env::temp_dir().to_string_lossy().to_string();

    let temp_path = download_file(&app, &url, &temp_dir).await?;

    unzip_file(&app, &temp_path, &target_dir).await?;

    if let Err(e) = std::fs::remove_file(&temp_path) {
        return Err(format!("Failed to remove temp file: {}", e));
    }

    Ok(())
}
