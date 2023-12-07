#![forbid(unsafe_code)]

mod kdbush;

use kdbush::KDBush;
use serde::{Deserialize, Serialize};
use std::f64::{consts::PI, INFINITY};

/// An offset index used to access the zoom level value associated with a cluster in the data arrays
const OFFSET_ZOOM: usize = 2;

/// An offset index used to access the ID associated with a cluster in the data arrays
const OFFSET_ID: usize = 3;

/// An offset index used to access the identifier of the parent cluster of a point in the data arrays
const OFFSET_PARENT: usize = 4;

/// An offset index used to access the number of points contained within a cluster at the given zoom level in the data arrays
const OFFSET_NUM: usize = 5;

/// An offset index used to access the properties associated with a cluster in the data arrays
const OFFSET_PROP: usize = 6;

/// Supercluster configuration options
#[derive(Clone, Debug)]
pub struct Options {
    /// Min zoom level to generate clusters on
    pub min_zoom: u8,

    /// Max zoom level to cluster the points on
    pub max_zoom: u8,

    /// Minimum points to form a cluster
    pub min_points: u8,

    /// Cluster radius in pixels
    pub radius: f64,

    /// Tile extent (radius is calculated relative to it)
    pub extent: f64,

    /// Size of the KD-tree leaf node, affects performance
    pub node_size: usize,
}

/// GeoJSON Point
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    /// Point type
    #[serde(rename = "type")]
    pub r#type: String,

    /// Array of coordinates with longitude as first value and latitude as second one
    pub coordinates: Vec<f64>,
}

/// Feature metadata
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Properties {
    /// Feature's name
    pub name: Option<String>,

    /// Indicates whether the entity is a cluster
    pub cluster: Option<bool>,

    /// Cluster's unique identifier
    pub cluster_id: Option<usize>,

    // Number of points within a cluster
    pub point_count: Option<usize>,

    /// An abbreviated point count, useful for display
    pub point_count_abbreviated: Option<String>,
}

/// A GeoJSON Feature<Point>
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    /// Feature type
    #[serde(rename = "type")]
    pub r#type: String,

    /// Feature ID
    pub id: Option<usize>,

    /// Feature metadata
    pub properties: Properties,

    /// Geometry of the feature
    pub geometry: Option<Geometry>,
}

/// Collection of GeoJSON features in a specific tile
#[derive(Debug, Serialize, Deserialize)]
pub struct Tile {
    /// GeoJSON features
    pub features: Vec<Feature>,
}

#[derive(Clone, Debug)]
/// A spatial clustering configuration and data structure
pub struct Supercluster {
    /// Configuration settings
    options: Options,

    /// Vector of KDBush structures for different zoom levels
    trees: Vec<KDBush>,

    /// Stride used for data access within the KD-tree
    stride: usize,

    /// Input data points
    pub points: Vec<Feature>,

    /// Clusters metadata
    cluster_props: Vec<Properties>,
}

impl Supercluster {
    /// Create a new instance of `Supercluster` with the specified configuration settings.
    ///
    /// # Arguments
    ///
    /// - `options`: The configuration options for Supercluster.
    ///
    /// # Returns
    ///
    /// A new `Supercluster` instance with the given configuration.
    pub fn new(options: Options) -> Self {
        let capacity = options.max_zoom + 1;
        let trees: Vec<KDBush> = (0..capacity + 1)
            .map(|_| KDBush::new(0, options.node_size))
            .collect();

        Supercluster {
            trees,
            options,
            stride: 6,
            points: vec![],
            cluster_props: vec![],
        }
    }

    /// Load the input GeoJSON points into the Supercluster instance, performing clustering at various zoom levels.
    ///
    /// # Arguments
    ///
    /// - `points`: A vector of GeoJSON features representing input points to be clustered.
    ///
    /// # Returns
    ///
    /// A mutable reference to the updated `Supercluster` instance.
    pub fn load(&mut self, points: Vec<Feature>) -> &mut Self {
        let min_zoom = self.options.min_zoom;
        let max_zoom = self.options.max_zoom;

        self.points = points;

        // Generate a cluster object for each point and index input points into a KD-tree
        let mut data = vec![];

        for (i, p) in self.points.iter().enumerate() {
            if p.geometry.is_none() {
                continue;
            }

            let coordinates = &p.geometry.as_ref().unwrap().coordinates;

            // Store internal point/cluster data in flat numeric arrays for performance
            // Longitude
            data.push(lng_x(coordinates[0]));

            // Latitude
            data.push(lat_y(coordinates[1]));

            // The last zoom the point was processed at
            data.push(INFINITY);

            // Index of the source feature in the original input array
            data.push(i as f64);

            // Parent cluster id
            data.push(-1.0);

            // Number of points in a cluster
            data.push(1.0);
        }

        self.trees[(max_zoom as usize) + 1] = self.create_tree(data);

        // Cluster points on max zoom, then cluster the results on previous zoom, etc.;
        // Results in a cluster hierarchy across zoom levels
        for zoom in (min_zoom..=max_zoom).rev() {
            // Create a new set of clusters for the zoom and index them with a KD-tree
            let (previous, current) = self.cluster(&self.trees[(zoom as usize) + 1], zoom);

            self.trees[(zoom as usize) + 1].data = previous;
            self.trees[zoom as usize] = self.create_tree(current);
        }

        self
    }

    /// Retrieve clustered features within the specified bounding box and zoom level.
    ///
    /// # Arguments
    ///
    /// - `bbox`: The bounding box as an array of four coordinates [min_lng, min_lat, max_lng, max_lat].
    /// - `zoom`: The zoom level at which to retrieve clusters.
    ///
    /// # Returns
    ///
    /// A vector of GeoJSON features representing the clusters within the specified bounding box and zoom level.
    pub fn get_clusters(&self, bbox: [f64; 4], zoom: u8) -> Vec<Feature> {
        let mut min_lng = ((((bbox[0] + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;
        let min_lat = f64::max(-90.0, f64::min(90.0, bbox[1]));
        let mut max_lng = if bbox[2] == 180.0 {
            180.0
        } else {
            ((((bbox[2] + 180.0) % 360.0) + 360.0) % 360.0) - 180.0
        };
        let max_lat = f64::max(-90.0, f64::min(90.0, bbox[3]));

        if bbox[2] - bbox[0] >= 360.0 {
            min_lng = -180.0;
            max_lng = 180.0;
        } else if min_lng > max_lng {
            let eastern_hem = self.get_clusters([min_lng, min_lat, 180.0, max_lat], zoom);
            let western_hem = self.get_clusters([-180.0, min_lat, max_lng, max_lat], zoom);

            return eastern_hem.into_iter().chain(western_hem).collect();
        }

        let tree = &self.trees[self.limit_zoom(zoom)];
        let ids = tree.range(
            lng_x(min_lng),
            lat_y(max_lat),
            lng_x(max_lng),
            lat_y(min_lat),
        );
        let mut clusters = Vec::new();

        for id in ids {
            let k = self.stride * id;

            clusters.push(if tree.data[k + OFFSET_NUM] > 1.0 {
                get_cluster_json(&tree.data, k, &self.cluster_props)
            } else {
                self.points[tree.data[k + OFFSET_ID] as usize].clone()
            });
        }

        clusters
    }

    /// Retrieve the cluster features for a specified cluster ID.
    ///
    /// # Arguments
    ///
    /// - `cluster_id`: The unique identifier of the cluster.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of GeoJSON features representing the children of the specified cluster if successful,
    /// or an error message if the cluster is not found.
    pub fn get_children(&self, cluster_id: usize) -> Result<Vec<Feature>, &'static str> {
        let origin_id = self.get_origin_id(cluster_id);
        let origin_zoom = self.get_origin_zoom(cluster_id);
        let error_msg = "No cluster with the specified id.";
        let tree = self.trees.get(origin_zoom);

        if tree.is_none() {
            return Err(error_msg);
        }

        let tree = tree.expect("tree is not defined");
        let data = &tree.data;

        if origin_id * self.stride >= data.len() {
            return Err(error_msg);
        }

        let r = self.options.radius
            / (self.options.extent * f64::powf(2.0, (origin_zoom as f64) - 1.0));

        let x = data[origin_id * self.stride];
        let y = data[origin_id * self.stride + 1];

        let ids = tree.within(x, y, r);

        let mut children = Vec::new();

        for id in ids {
            let k = id * self.stride;

            if data[k + OFFSET_PARENT] == (cluster_id as f64) {
                if data[k + OFFSET_NUM] > 1.0 {
                    children.push(get_cluster_json(data, k, &self.cluster_props));
                } else {
                    let point_id = data[k + OFFSET_ID] as usize;

                    children.push(self.points[point_id].clone());
                }
            }
        }

        if children.is_empty() {
            return Err(error_msg);
        }

        Ok(children)
    }

    /// Retrieve individual leaf features within a cluster.
    ///
    /// # Arguments
    ///
    /// - `cluster_id`: The unique identifier of the cluster.
    /// - `limit`: The maximum number of leaf features to retrieve.
    /// - `offset`: The offset to start retrieving leaf features.
    ///
    /// # Returns
    ///
    /// A vector of GeoJSON features representing the individual leaf features within the cluster.
    pub fn get_leaves(&self, cluster_id: usize, limit: usize, offset: usize) -> Vec<Feature> {
        let mut leaves = vec![];

        self.append_leaves(&mut leaves, cluster_id, limit, offset, 0);

        leaves
    }

    /// Retrieve a vector of features within a tile at the given zoom level and tile coordinates.
    ///
    /// # Arguments
    ///
    /// - `z`: The zoom level of the tile.
    /// - `x`: The X coordinate of the tile.
    /// - `y`: The Y coordinate of the tile.
    ///
    /// # Returns
    ///
    /// An optional `Tile` containing a vector of GeoJSON features within the specified tile, or `None` if there are no features.
    pub fn get_tile(&self, z: u8, x: f64, y: f64) -> Option<Tile> {
        let tree = &self.trees[self.limit_zoom(z)];
        let z2: f64 = (2u32).pow(z as u32) as f64;
        let p = self.options.radius / self.options.extent;
        let top = (y - p) / z2;
        let bottom = (y + 1.0 + p) / z2;

        let mut tile = Tile { features: vec![] };

        let ids = tree.range((x - p) / z2, top, (x + 1.0 + p) / z2, bottom);

        self.add_tile_features(&ids, &tree.data, x, y, z2, &mut tile);

        if x == 0.0 {
            let ids = tree.range(1.0 - p / z2, top, 1.0, bottom);

            self.add_tile_features(&ids, &tree.data, z2, y, z2, &mut tile);
        }

        if x == z2 - 1.0 {
            let ids = tree.range(0.0, top, p / z2, bottom);

            self.add_tile_features(&ids, &tree.data, -1.0, y, z2, &mut tile);
        }

        if tile.features.is_empty() {
            None
        } else {
            Some(tile)
        }
    }

    /// Determine the zoom level at which a specific cluster expands.
    ///
    /// # Arguments
    ///
    /// - `cluster_id`: The unique identifier of the cluster.
    ///
    /// # Returns
    ///
    /// The zoom level at which the cluster expands.
    pub fn get_cluster_expansion_zoom(&self, mut cluster_id: usize) -> usize {
        let mut expansion_zoom = self.get_origin_zoom(cluster_id) - 1;

        while expansion_zoom <= (self.options.max_zoom as usize) {
            let children = if self.get_children(cluster_id).is_ok() {
                self.get_children(cluster_id).unwrap()
            } else {
                break;
            };

            expansion_zoom += 1;

            if children.len() != 1 {
                break;
            }

            cluster_id = if children[0].properties.cluster_id.is_some() {
                children[0].properties.cluster_id.unwrap()
            } else {
                break;
            };
        }

        expansion_zoom
    }

    /// Appends leaves (features) to the result vector based on the specified criteria.
    ///
    /// # Arguments
    ///
    /// - `result`: A mutable reference to a vector where leaves will be appended.
    /// - `cluster_id`: The identifier of the cluster whose leaves are being collected.
    /// - `limit`: The maximum number of leaves to collect.
    /// - `offset`: The number of leaves to skip before starting to collect.
    /// - `skipped`: The current count of skipped leaves, used for tracking the progress.
    ///
    /// # Returns
    ///
    /// The updated count of skipped leaves after processing the current cluster.
    fn append_leaves(
        &self,
        result: &mut Vec<Feature>,
        cluster_id: usize,
        limit: usize,
        offset: usize,
        mut skipped: usize,
    ) -> usize {
        let cluster = self.get_children(cluster_id).unwrap();

        for child in cluster {
            if child.properties.cluster.is_some() {
                if skipped + child.properties.point_count.unwrap() <= offset {
                    // Skip the whole cluster
                    skipped += child.properties.point_count.unwrap();
                } else {
                    // Enter the cluster
                    skipped = self.append_leaves(
                        result,
                        child.properties.cluster_id.unwrap(),
                        limit,
                        offset,
                        skipped,
                    );
                    // Exit the cluster
                }
            } else if skipped < offset {
                // Skip a single point
                skipped += 1;
            } else {
                // Add a single point
                result.push(child);
            }

            if result.len() == limit {
                break;
            }
        }

        skipped
    }

    /// Create a KD-tree using the specified data, which is used for spatial indexing.
    ///
    /// # Arguments
    ///
    /// - `data`: A vector of flat numeric arrays representing point data for the KD-tree.
    ///
    /// # Returns
    ///
    /// A `KDBush` instance with the specified data.
    fn create_tree(&mut self, data: Vec<f64>) -> KDBush {
        let mut tree = KDBush::new(data.len() / self.stride, self.options.node_size);

        for i in (0..data.len()).step_by(self.stride) {
            tree.add_point(data[i], data[i + 1]);
        }

        tree.build_index();
        tree.data = data;

        tree
    }

    /// Populate a tile with features based on the specified point IDs, data, and tile parameters.
    ///
    /// # Arguments
    ///
    /// - `ids`: A vector of point IDs used for populating the tile.
    /// - `data`: A reference to the flat numeric arrays representing point data.
    /// - `x`: The X coordinate of the tile.
    /// - `y`: The Y coordinate of the tile.
    /// - `z2`: The zoom level multiplied by 2.
    /// - `tile`: A mutable reference to the `Tile` to be populated with features.
    fn add_tile_features(
        &self,
        ids: &Vec<usize>,
        data: &[f64],
        x: f64,
        y: f64,
        z2: f64,
        tile: &mut Tile,
    ) {
        for i in ids {
            let k = i * self.stride;
            let is_cluster = data[k + OFFSET_NUM] > 1.0;

            let px;
            let py;
            let properties;

            if is_cluster {
                properties = get_cluster_properties(data, k, &self.cluster_props);

                px = data[k];
                py = data[k + 1];
            } else {
                let p = &self.points[data[k + OFFSET_ID] as usize];
                properties = p.properties.clone();

                let coordinates = &p.geometry.as_ref().unwrap().coordinates;
                px = lng_x(coordinates[0]);
                py = lat_y(coordinates[1]);
            }

            let id = if is_cluster {
                Some(data[k + OFFSET_ID] as usize)
            } else {
                self.points[data[k + OFFSET_ID] as usize].id
            };

            tile.features.push(Feature {
                id,
                properties,
                r#type: "Feature".to_string(),
                geometry: Some(Geometry {
                    r#type: "Point".to_string(),
                    coordinates: vec![
                        (self.options.extent * (px * z2 - x)).round(),
                        (self.options.extent * (py * z2 - y)).round(),
                    ],
                }),
            });
        }
    }

    /// Calculate the effective zoom level that takes into account the configured minimum and maximum zoom levels.
    ///
    /// # Arguments
    ///
    /// - `zoom`: The initial zoom level.
    ///
    /// # Returns
    ///
    /// The effective zoom level considering the configured minimum and maximum zoom levels.
    fn limit_zoom(&self, zoom: u8) -> usize {
        zoom.max(self.options.min_zoom)
            .min(self.options.max_zoom + 1) as usize
    }

    /// Cluster points on a given zoom level using a KD-tree and returns updated data arrays.
    ///
    /// # Arguments
    ///
    /// - `tree`: A reference to the KD-tree structure for spatial indexing.
    /// - `zoom`: The zoom level at which clustering is performed.
    ///
    /// # Returns
    ///
    /// A tuple of two vectors: the first one contains updated data arrays for the current zoom level,
    /// and the second one contains data arrays for the next zoom level.
    fn cluster(&self, tree: &KDBush, zoom: u8) -> (Vec<f64>, Vec<f64>) {
        let r = self.options.radius / (self.options.extent * (2.0_f64).powi(zoom as i32));
        let mut data = tree.data.clone();
        let mut next_data = Vec::new();

        // Loop through each point
        for i in (0..data.len()).step_by(self.stride) {
            // If we've already visited the point at this zoom level, skip it
            if data[i + OFFSET_ZOOM] <= (zoom as f64) {
                continue;
            }

            data[i + OFFSET_ZOOM] = zoom as f64;

            // Find all nearby points
            let x = data[i];
            let y = data[i + 1];

            let neighbor_ids = tree.within(x, y, r);

            let num_points_origin = data[i + OFFSET_NUM];
            let mut num_points = num_points_origin;

            // Count the number of points in a potential cluster
            for neighbor_id in &neighbor_ids {
                let k = neighbor_id * self.stride;

                // Filter out neighbors that are already processed
                if data[k + OFFSET_ZOOM] > (zoom as f64) {
                    num_points += data[k + OFFSET_NUM];
                }
            }

            // If there were neighbors to merge, and there are enough points to form a cluster
            if num_points > num_points_origin && num_points >= (self.options.min_points as f64) {
                let mut wx = x * num_points_origin;
                let mut wy = y * num_points_origin;

                // Encode both zoom and point index on which the cluster originated -- offset by total length of features
                let id = ((i / self.stride) << 5) + ((zoom as usize) + 1) + self.points.len();

                for neighbor_id in neighbor_ids {
                    let k = neighbor_id * self.stride;

                    if data[k + OFFSET_ZOOM] <= (zoom as f64) {
                        continue;
                    }

                    // Save the zoom (so it doesn't get processed twice)
                    data[k + OFFSET_ZOOM] = zoom as f64;

                    let num_points2 = data[k + OFFSET_NUM];

                    // Accumulate coordinates for calculating weighted center
                    wx += data[k] * num_points2;
                    wy += data[k + 1] * num_points2;

                    data[k + OFFSET_PARENT] = id as f64;
                }

                data[i + OFFSET_PARENT] = id as f64;

                next_data.push(wx / num_points);
                next_data.push(wy / num_points);
                next_data.push(INFINITY);
                next_data.push(id as f64);
                next_data.push(-1.0);
                next_data.push(num_points);
            } else {
                // Left points as unclustered
                for j in 0..self.stride {
                    next_data.push(data[i + j]);
                }

                if num_points > 1.0 {
                    for neighbor_id in neighbor_ids {
                        let k = neighbor_id * self.stride;

                        if data[k + OFFSET_ZOOM] <= (zoom as f64) {
                            continue;
                        }

                        data[k + OFFSET_ZOOM] = zoom as f64;

                        for j in 0..self.stride {
                            next_data.push(data[k + j]);
                        }
                    }
                }
            }
        }

        (data, next_data)
    }

    /// Get the index of the point from which the cluster originated.
    ///
    /// # Arguments
    ///
    /// - `cluster_id`: The unique identifier of the cluster.
    ///
    /// # Returns
    ///
    /// The index of the point from which the cluster originated.
    fn get_origin_id(&self, cluster_id: usize) -> usize {
        (cluster_id - self.points.len()) >> 5
    }

    /// Get the zoom of the point from which the cluster originated.
    ///
    /// # Arguments
    ///
    /// - `cluster_id`: The unique identifier of the cluster.
    ///
    /// # Returns
    ///
    /// The zoom level of the point from which the cluster originated.
    fn get_origin_zoom(&self, cluster_id: usize) -> usize {
        (cluster_id - self.points.len()) % 32
    }
}

/// Convert clustered point data into a GeoJSON feature representing a cluster.
///
/// # Arguments
///
/// - `data`: A reference to the flat numeric arrays representing point data.
/// - `i`: The index in the data array for the cluster.
/// - `cluster_props`: A reference to a vector of cluster properties.
///
/// # Returns
///
/// A GeoJSON feature representing a cluster.
fn get_cluster_json(data: &[f64], i: usize, cluster_props: &[Properties]) -> Feature {
    Feature {
        r#type: "Feature".to_string(),
        id: Some(data[i + OFFSET_ID] as usize),
        properties: get_cluster_properties(data, i, cluster_props),
        geometry: Some(Geometry {
            r#type: "Point".to_string(),
            coordinates: vec![x_lng(data[i]), y_lat(data[i + 1])],
        }),
    }
}

/// Retrieve properties for a cluster based on clustered point data.
///
/// # Arguments
///
/// - `data`: A reference to the flat numeric arrays representing point data.
/// - `i`: The index in the data array for the cluster.
/// - `cluster_props`: A reference to a vector of cluster properties.
///
/// # Returns
///
/// Properties for the cluster based on the clustered point data.
fn get_cluster_properties(data: &[f64], i: usize, cluster_props: &[Properties]) -> Properties {
    let count = data[i + OFFSET_NUM];
    let abbrev = if count >= 10000.0 {
        format!("{}k", count / 1000.0)
    } else if count >= 1000.0 {
        format!("{:}k", count / 100.0 / 10.0)
    } else {
        count.to_string()
    };

    let mut properties = if !cluster_props.is_empty() && data.get(i + OFFSET_PROP).is_some() {
        cluster_props[data[i + OFFSET_PROP] as usize].clone()
    } else {
        Properties::default()
    };

    properties.cluster = Some(true);
    properties.cluster_id = Some(data[i + OFFSET_ID] as usize);
    properties.point_count = Some(count as usize);
    properties.point_count_abbreviated = Some(abbrev);

    properties
}

/// Convert longitude to spherical mercator in the [0..1] range.
///
/// # Arguments
///
/// - `lng`: The longitude value to be converted.
///
/// # Returns
///
/// The converted value in the [0..1] range.
fn lng_x(lng: f64) -> f64 {
    lng / 360.0 + 0.5
}

/// Convert latitude to spherical mercator in the [0..1] range.
///
/// # Arguments
///
/// - `lat`: The latitude value to be converted.
///
/// # Returns
///
/// The converted value in the [0..1] range.
fn lat_y(lat: f64) -> f64 {
    let sin = lat.to_radians().sin();
    let y = 0.5 - (0.25 * ((1.0 + sin) / (1.0 - sin)).ln()) / PI;

    if y < 0.0 {
        0.0
    } else if y > 1.0 {
        1.0
    } else {
        y
    }
}

/// Convert spherical mercator to longitude.
///
/// # Arguments
///
/// - `x`: The spherical mercator value to be converted.
///
/// # Returns
///
/// The converted longitude value.
fn x_lng(x: f64) -> f64 {
    (x - 0.5) * 360.0
}

/// Convert spherical mercator to latitude.
///
/// # Arguments
///
/// - `y`: The spherical mercator value to be converted.
///
/// # Returns
///
/// The converted latitude value.
fn y_lat(y: f64) -> f64 {
    let y2 = ((180.0 - y * 360.0) * PI) / 180.0;
    (360.0 * y2.exp().atan()) / PI - 90.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Supercluster {
        Supercluster::new(Options {
            radius: 40.0,
            extent: 512.0,
            max_zoom: 16,
            min_zoom: 0,
            min_points: 2,
            node_size: 64,
        })
    }

    #[test]
    fn test_limit_zoom() {
        let supercluster = setup();

        assert_eq!(supercluster.limit_zoom(5), 5);
    }

    #[test]
    fn test_get_origin_id() {
        let supercluster = setup();

        assert_eq!(supercluster.get_origin_id(100), 3);
    }

    #[test]
    fn test_get_origin_zoom() {
        let supercluster = setup();

        assert_eq!(supercluster.get_origin_zoom(100), 4);
    }

    #[test]
    fn test_get_cluster_json_with_cluster_props() {
        let data = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let i = 0;
        let cluster_props = vec![Properties {
            cluster: Some(false),
            cluster_id: Some(0),
            point_count: Some(0),
            name: Some("name".to_string()),
            point_count_abbreviated: Some("0".to_string()),
        }];

        let result = get_cluster_json(&data, i, &cluster_props);

        assert_eq!(result.r#type, "Feature".to_string());
        assert_eq!(result.id, Some(0));
        assert_eq!(
            result.geometry.as_ref().unwrap().r#type,
            "Point".to_string()
        );
        assert_eq!(
            result.geometry.unwrap().coordinates,
            vec![-180.0, 85.05112877980659]
        );

        let properties = result.properties;

        assert_eq!(properties.cluster, Some(true));
        assert_eq!(properties.cluster_id, Some(0));
        assert_eq!(properties.point_count, Some(3));
        assert_eq!(properties.name, Some("name".to_string()));
        assert_eq!(properties.point_count_abbreviated, Some("3".to_string()));
    }

    #[test]
    fn test_get_cluster_json_without_cluster_props() {
        let data = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let i = 0;
        let cluster_props = vec![];

        let result = get_cluster_json(&data, i, &cluster_props);

        assert_eq!(result.id, Some(0));
        assert_eq!(result.r#type, "Feature".to_string());
        assert_eq!(
            result.geometry.as_ref().unwrap().r#type,
            "Point".to_string()
        );
        assert_eq!(
            result.geometry.unwrap().coordinates,
            vec![-180.0, 85.05112877980659]
        );

        let properties = result.properties;

        assert!(properties.name.is_none());
        assert_eq!(properties.cluster, Some(true));
        assert_eq!(properties.cluster_id, Some(0));
        assert_eq!(properties.point_count, Some(3));
        assert_eq!(properties.point_count_abbreviated, Some("3".to_string()));
    }

    #[test]
    fn test_get_cluster_properties_with_cluster_props() {
        let data = [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0, 0.0];
        let i = 0;
        let cluster_props = vec![Properties {
            cluster: Some(false),
            cluster_id: Some(0),
            point_count: Some(0),
            name: Some("name".to_string()),
            point_count_abbreviated: Some("0".to_string()),
        }];

        let result = get_cluster_properties(&data, i, &cluster_props);

        assert_eq!(result.cluster, Some(true));
        assert_eq!(result.cluster_id, Some(0));
        assert_eq!(result.point_count, Some(10000));
        assert_eq!(result.name, Some("name".to_string()));
        assert_eq!(result.point_count_abbreviated, Some("10k".to_string()));
    }

    #[test]
    fn test_get_cluster_properties_without_cluster_props() {
        let data = [0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0];
        let i = 0;
        let cluster_props = vec![];

        let result = get_cluster_properties(&data, i, &cluster_props);

        assert!(result.name.is_none());
        assert_eq!(result.cluster, Some(true));
        assert_eq!(result.cluster_id, Some(0));
        assert_eq!(result.point_count, Some(1000));
        assert_eq!(result.point_count_abbreviated, Some("1k".to_string()));
    }

    #[test]
    fn test_lng_x() {
        assert_eq!(lng_x(0.0), 0.5);
        assert_eq!(lng_x(180.0), 1.0);
        assert_eq!(lng_x(-180.0), 0.0);
        assert_eq!(lng_x(90.0), 0.75);
        assert_eq!(lng_x(-90.0), 0.25);
    }

    #[test]
    fn test_lat_y() {
        assert_eq!(lat_y(0.0), 0.5);
        assert_eq!(lat_y(90.0), 0.0);
        assert_eq!(lat_y(-90.0), 1.0);
        assert_eq!(lat_y(45.0), 0.35972503691520497);
        assert_eq!(lat_y(-45.0), 0.640274963084795);
    }

    #[test]
    fn test_x_lng() {
        assert_eq!(x_lng(0.5), 0.0);
        assert_eq!(x_lng(1.0), 180.0);
        assert_eq!(x_lng(0.0), -180.0);
        assert_eq!(x_lng(0.75), 90.0);
        assert_eq!(x_lng(0.25), -90.0);
    }

    #[test]
    fn test_y_lat() {
        assert_eq!(y_lat(0.5), 0.0);
        assert_eq!(y_lat(0.875), -79.17133464081944);
        assert_eq!(y_lat(0.125), 79.17133464081945);
    }
}
