use std::{fs, path::Path};
use supercluster::{Feature, Options, Tile};

#[allow(dead_code)]
pub fn get_options(radius: f64, extent: f64, min_points: i32, max_zoom: i32) -> Options {
    Options {
        radius,
        extent,
        max_zoom,
        min_zoom: 0,
        min_points,
        node_size: 64,
    }
}

#[allow(dead_code)]
pub fn load_places() -> Vec<Feature> {
    let file_path = Path::new("./tests/common/places.json");
    let json_string = fs::read_to_string(file_path).expect("places.json was not found");

    serde_json::from_str(&json_string).expect("places.json was not parsed")
}

#[allow(dead_code)]
pub fn load_tile_places() -> Tile {
    let file_path = Path::new("./tests/common/places-tile-0-0-0.json");
    let json_string = fs::read_to_string(file_path).expect("places-tile-0-0-0.json was not found");

    serde_json::from_str(&json_string).expect("places-tile-0-0-0.json was not parsed")
}

#[allow(dead_code)]
pub fn load_tile_places_with_min_5() -> Tile {
    let file_path = Path::new("./tests/common/places-tile-0-0-0-min-5.json");
    let json_string =
        fs::read_to_string(file_path).expect("places-tile-0-0-0-min-5.json was not found");

    serde_json::from_str(&json_string).expect("places-z0-0-0-min5.json was not parsed")
}
