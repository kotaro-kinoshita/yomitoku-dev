from .utils.misc import load_config
from .data.functions import load_image
from .modules import BaseModule
from .modules import Detection
from .modules import Recognition


class OCR(BaseModule):
    def __init__(self, path_cfg, visualize=True):
        self.cfg = load_config(path_cfg)
        self.detector = Detection(self.cfg, visualize=visualize, device=self.cfg.DEVICE)
        self.recognizer = Recognition(
            self.cfg, visualize=visualize, device=self.cfg.DEVICE
        )

    def format(self, quads, det_scores, contents, rec_scores):
        words = []
        for quad, det_score, pred, rec_score in zip(
            quads, det_scores, contents, rec_scores
        ):
            words.append(
                {
                    "points": quad,
                    "content": pred,
                    "det_score": det_score,
                    "rec_score": rec_score,
                }
            )
        return words

    def __call__(self, path_img):
        img = load_image(path_img)
        h, w = img.shape[:2]
        quads, det_scores, vis = self.detector(img)
        contents, rec_scores, vis = self.recognizer(img, quads, vis=vis)

        results = {}
        results.update(image_size=(w, h))
        results.update(words=self.format(quads, det_scores, contents, rec_scores))
        return results, vis
