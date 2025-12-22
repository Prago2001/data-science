from PIL import Image
import numpy as np
import time
from typing import Literal
from scipy.sparse import csc_matrix, find


class KMeansImpl:
    def __init__(self):
        pass

    def load_image(self, image_name="1.jpeg"):
        """
        Returns the image numpy array.
        It is important that image_name parameter defaults to the choice image name.
        """
        self.pixels = np.array(Image.open(image_name).convert("RGB"))
        print(f"Shape of orignal image: {self.pixels.shape}")
        return self.pixels

    def change_image_shape(
        self,
        pixels: np.ndarray,
        type: Literal["flat", "orignal"],
        orignal_size: None | tuple = None,
    ):
        if type == "flat":
            return np.reshape(
                pixels, (pixels.shape[0] * pixels.shape[1], pixels.shape[2])
            )
        elif type == "orignal" and orignal_size is not None:
            return np.reshape(pixels, orignal_size, order="C")

    def compute_distances(self, pixels: np.ndarray, means: np.ndarray, norm_distance=2):
        if norm_distance == 2:
            # pixels_norm = np.sum(pixels**2, axis=1, keepdims=True)
            centres_norm = np.sum(means**2, axis=1, keepdims=True).T
            pixels_cross_centres = pixels @ means.T
            return centres_norm - 2 * pixels_cross_centres
        elif norm_distance == 1:
            return np.sum(
                np.absolute(pixels[:, np.newaxis, :] - means[np.newaxis, :, :]), axis=2
            )

    def update_centres(
        self, pixels: np.ndarray, labels: np.ndarray, k: int, norm_distance=2
    ):
        pixel_length, channels = pixels.shape
        if norm_distance == 2:
            sparse = csc_matrix(
                (np.ones(pixel_length), (np.arange(0, pixel_length, 1), labels)),
                shape=(pixel_length, k),
            )
            try:
                count = sparse.sum(axis=0)
                centres = np.array((sparse.T.dot(pixels)) / count)
                return centres, centres.shape[0]
            except Exception as e:
                pass
            finally:
                valid_centres = []
                for cluster_number in range(k):
                    idx = find(sparse[:, cluster_number])[0]
                    num_points = idx.shape[0]
                    if num_points == 0:
                        continue
                    else:
                        centre = np.array(
                            sparse[:, cluster_number].T.dot(pixels) / float(num_points)
                        )[0, :]
                        valid_centres.append(centre)
                valid_centres = np.array(valid_centres)
                return valid_centres, valid_centres.shape[0]
        else:
            centroids = []
            sparse = csc_matrix(
                (np.ones(pixel_length), (np.arange(0, pixel_length, 1), labels)),
                shape=(pixel_length, k),
            )
            for cluster_number in range(k):
                idx = find(sparse[:, cluster_number])[0]
                num_points = idx.shape[0]
                if num_points != 0:
                    centroids.append(np.median(idx, axis=0))
            return np.array(centroids), len(centroids)

    def wcss(self, distances: np.ndarray):
        return np.sum(np.amin(distances, axis=1))

    def check_convergance(
        self, old_centres: np.ndarray, centres: np.ndarray, norm_distance=2
    ):
        diff = np.linalg.norm(centres - old_centres, ord="fro")
        # print(diff)
        if diff > 1e-6:
            return False
        else:
            return True

    def compress(
        self,
        num_clusters,
        norm_distance=2,
    ):
        """
        Compress the image using K-Means clustering.

        Parameters:
            num_clusters: Number of clusters (k) to use for compression.
            norm_distance: Type of distance metric to use for clustering.
                            Can be 1 for Manhattan distance or 2 for Euclidean distance.
                            Default is 2 (Euclidean).

        Returns:
            Dictionary containing:
                "class": Cluster assignments for each pixel.
                "centroid": Locations of the cluster centroids.
                "img": Compressed image with each pixel assigned to its closest cluster.
                "number_of_iterations": total iterations taken by algorithm
                "time_taken": time taken by the compression algorithm
        """
        print(f"\nStarting compression for k={num_clusters}")
        start_time = time.time()
        orignal_shape = self.pixels.shape
        flat_image = self.change_image_shape(self.pixels, "flat").astype(
            np.float32, copy=False
        )
        centres = flat_image[
            np.random.choice(flat_image.shape[0], num_clusters, replace=False)
        ]

        converged = False
        iterations = 0

        labels = np.zeros(shape=flat_image.shape)
        previous_wcss = np.inf
        wcss = np.inf

        while converged is False:
            previous_wcss = wcss
            iterations += 1
            iter_start_time = time.time()
            old_centres = centres.copy()
            distances = self.compute_distances(
                flat_image, centres, norm_distance=norm_distance
            )
            # print(distances.shape)
            labels = np.argmin(distances, axis=1)
            # print(labels.shape)
            centres, num_clusters = self.update_centres(
                flat_image, labels, num_clusters
            )
            wcss = self.wcss(distances)

            converged = self.check_convergance(old_centres[:num_clusters], centres)
            # print(
            #     f"Completed iteration {iterations} in {time.time() - iter_start_time}\t\tWCSS = {np.round(wcss,3)}"
            # )
            if wcss - previous_wcss >= 0:
                break

        compressed_img_flat = centres[labels]
        compressed_image = self.change_image_shape(
            compressed_img_flat, "orignal", orignal_shape
        )

        elapsed_time = time.time() - start_time
        print(f"Completed in {round(elapsed_time,3)} seconds..")
        map = {
            "class": labels + 1,
            "centroid": centres,
            "img": compressed_image,
            "number_of_iterations": iterations,
            "time_taken": elapsed_time,
            "additional_args": {"wcss": wcss},
        }
        return map
