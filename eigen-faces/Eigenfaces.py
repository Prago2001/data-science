import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class EigenFaces:
    """
    Create PCA model for each subject and store it's mean component and eigen vectors
    """

    def __init__(self):
        self.db = {}

    def add_subject(self, subject_id: int, images: np.ndarray, n_components: int = 6):
        """
        Add a subject to the db with `subject_id` and `images`

        :param subject_id:
        :type subject_id: int
        :param images:
        :type images: np.ndarray
        :param n_components Number of components to keep in PCA
        :type int
        """
        if images.shape[0] < n_components:
            n_components = images.shape[0]
            print(f"Using {images.shape[0]} as the number of components in PCA")

        pca = PCA(n_components=n_components, whiten=True, svd_solver="full")

        pca.fit(images)

        self.db[subject_id] = {"mean": pca.mean_, "eigenfaces": pca.components_}

        print(f"Added eigenfaces of subject {subject_id}...")

    def plot_eigen_faces(self, subject_id: int):
        if subject_id in self.db:
            subject_data = self.db[subject_id]
            fig, axes = plt.subplots(3, 2, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(subject_data["eigenfaces"][i].reshape(64, 64), cmap="gray")
                ax.set_axis_off()
                ax.set_title(f"Eigen Face {i+1}")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.suptitle(f"Subject {subject_id} Eigenfaces")
            plt.show()
        else:
            raise Exception(f"Subject ID {subject_id} not found...")

    def recognize(self, test_image: np.ndarray):
        """
        Identify which subject a test image belongs to.

        Args:
            test_image: np.ndarray of shape (n_features,)

        Returns:
            (subject_id: int, residual_dict: pd.Dataframe)
        """
        residuals = []

        for subject_id, data in self.db.items():
            mean = data["mean"]
            eigenfaces = data["eigenfaces"]

            # Center test image
            centered_test = test_image - mean

            # Project onto eigenfaces to get coefficients
            coeffs = centered_test.dot(eigenfaces.T)  # shape (6,)

            # Reconstruct from eigenspace
            reconstruction = coeffs.dot(eigenfaces)  # shape (n_pixels,)

            # Residual = distance from test to reconstruction
            residual = np.linalg.norm(centered_test - reconstruction) ** 2
            residuals.append({"Subject ID": subject_id, "Residual": residual})

        # Best match: lowest residual
        best_subject: int = min(residuals, key=lambda x: x["Residual"])
        return best_subject, pd.DataFrame(residuals).sort_values(
            by=["Residual"], ignore_index=True
        )
