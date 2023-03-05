import numpy as np
from nlp_project.utils import kldivergence, softmax, sampled_sphere

class Mahalanobis():

    def __init__(
        self,
        train_embeds_in: np.ndarray,
        test_embeds_in: np.ndarray,
        test_embeds_out: np.ndarray,
        train_labels_in: np.ndarray,
        substract_mean: bool = True,
        normalize_to_unity: bool = True,
        substract_train_distance: bool = True,
        norm_name: str = "L2"
    ) -> None:
        self.train_embeds_in = train_embeds_in
        self.test_embeds_in = test_embeds_in
        self.test_embeds_out = test_embeds_out
        self.train_labels_in = train_labels_in
        self.substract_mean = substract_mean
        self.normalize_to_unity = normalize_to_unity
        self.substract_train_distance = substract_train_distance
        self.norm_name = norm_name
        self.description = ""
        self.c = len(np.unique(self.train_labels_in))

    def __call__(self):
        self.pre_normalize()
        self.get_metrics()
        self.check_metrics()
        self.compute_distances()

        return self.onehots, self.scores

    def pre_normalize(self):
        """Normalize datasets"""
        all_train_mean = np.mean(self.train_embeds_in, axis=0, keepdims=True)

        if self.substract_mean:
            self.train_embeds_in -= all_train_mean
            self.test_embeds_in -= all_train_mean
            self.test_embeds_out -= all_train_mean
            self.description += " substract mean,"

        if self.normalize_to_unity:
            self.train_embeds_in = self.train_embeds_in / np.linalg.norm(self.train_embeds_in,axis=1,keepdims=True)
            self.test_embeds_in = self.test_embeds_in / np.linalg.norm(self.test_embeds_in,axis=1,keepdims=True)
            self.test_embeds_out = self.test_embeds_out / np.linalg.norm(self.test_embeds_out,axis=1,keepdims=True)
            self.description += " unit norm,"

    def get_metrics(self):
        """Get mean, cov matrix and inverse of cov matrix"""
        self.mean = np.mean(self.train_embeds_in,axis=0)
        self.cov = np.cov((self.train_embeds_in-(self.mean.reshape([1,-1]))).T)
        self.cov_inv = np.linalg.inv(self.cov)

        self.class_means = [np.mean(self.train_embeds_in[self.train_labels_in == c],axis=0) for c in range(self.c)]
        self.class_cov_invs = [np.cov((self.train_embeds_in[self.train_labels_in == c]-(self.class_means[c].reshape([1,-1]))).T) for c in range(self.c)]
        self.class_covs = [np.linalg.inv(self.class_cov_invs[c]) for c in range(self.c)]

    def check_metrics(self):
        """Check values for cov matrix and if cov_inv*cov give us identity matrix"""
        print(f"Average value of cov_inv matrix : {np.mean(self.cov_inv)}")
        print(f"Average distance between cov_inv*cov and identity matrix : {np.mean(np.abs(self.cov_inv.dot(self.cov) - np.eye(self.cov.shape[0])))}")

    def compute_distances(self):
        """Compute Mahalanobis distance for in and out datasets"""
        self.out_totrain = self._maha_distance(self.test_embeds_out, self.cov_inv, self.mean)
        self.in_totrain = self._maha_distance(self.test_embeds_in, self.cov_inv, self.mean)

        self.out_totrainclasses = [self._maha_distance(self.test_embeds_out, self.class_cov_invs[c], self.class_means[c]) for c in range(self.c)]
        self.in_totrainclasses = [self._maha_distance(self.test_embeds_in, self.class_cov_invs[c], self.class_means[c]) for c in range(self.c)]

        self.out_scores = np.min(np.stack(self.out_totrainclasses,axis=0),axis=0)
        self.in_scores = np.min(np.stack(self.in_totrainclasses,axis=0),axis=0)

        if self.substract_train_distance:
            self.out_scores -= self.out_totrain
            self.in_scores -= self.in_totrain

        self.onehots = np.array([1]*len(self.out_scores) + [0]*len(self.in_scores))
        self.scores = np.concatenate([self.out_scores, self.in_scores],axis=0)

    def _maha_distance(self, xs: np.ndarray, cov_inv: np.ndarray, mean: np.ndarray) -> np.array:
        diffs = xs - mean.reshape([1,-1])

        second_powers = np.matmul(diffs, cov_inv)*diffs

        if self.norm_name in [None,"L2"]:
            return np.sum(second_powers,axis=1)
        elif self.norm_name in ["L1"]:
            return np.sum(np.sqrt(np.abs(second_powers)),axis=1)
        elif self.norm_name in ["Linfty"]:
            return np.max(second_powers,axis=1)


class MaxSoftmax():
    def __init__(
        self,
        in_logits: np.ndarray,
        out_logits: np.ndarray,        
    ):
        self.in_logits = in_logits
        self.out_logits = out_logits

    def __call__(self):
        self.compute_scores()
        return self.onehots, self.scores
    
    def compute_scores(self):
        self.scores = np.array(
            np.concatenate([
                np.max(softmax(self.in_logits), axis=-1),
                np.max(softmax(self.out_logits), axis=-1),
            ], axis=0)
        )

        self.onehots = np.array(
            [1]*len(self.in_logits)+[0]*len(self.out_logits)
        )

class KLDivergence():
    def __init__(
        self,
        in_logits: np.ndarray,
        out_logits: np.ndarray,
    ):
        self.in_logits = in_logits
        self.out_logits = out_logits

    def __call__(self):
        self.compute_scores()
        return self.onehots, self.scores
 
    def compute_scores(self):
        
        self.scores = np.concatenate(
            [kldivergence(softmax(self.in_logits)), kldivergence(softmax(self.out_logits))]
        )

        self.onehots = np.array(
            [1]*len(self.in_logits)+[0]*len(self.out_logits)
        )  


class IRW():
    def __init__(
        self,
        in_train: np.ndarray,
        in_test: np.ndarray,
        out_test: np.ndarray,
        n_dirs: int = None
    ):
        self.in_train = in_train
        self.in_test = in_test
        self.out_test = out_test
        self.n_dirs = n_dirs

    def __call__(self):
        self.get_unit_sphere_vectors()
        self.get_unit_sphere_vectors()
        return self.onehots, self.scores

    def get_unit_sphere_vectors(self):
        if not self.n_dirs:
            self.n_dirs = 100 * self.in_train.shape[1]

        self.U = sampled_sphere(self.n_dirs, self.in_train.shape[1])

    def d_irw(self, x):
        return np.mean([
            np.min(
                np.mean(np.array([np.matmul(x_train - x, u) for x_train in self.in_train]) > 0), 
                np.mean(np.array([np.matmul(x_train - x, u) for x_train in self.in_train]) <= 0)
            )
            for u in self.U
        ])
    
    def compute_scores(self):
        self.scores = np.concatenate(
            [
                np.array([self.d_irw(x) for x in self.in_test]),
                np.array([self.d_irw(x) for x in self.out_test]),
            ]
        )

        self.onehots = np.array(
            [1]*len(self.in_test)+[0]*len(self.out_test)
        )  
    