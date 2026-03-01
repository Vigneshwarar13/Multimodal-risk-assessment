from typing import Dict, Optional, Tuple
import re
from pathlib import Path
import torch


class TamilCoercionClassifier:
    def __init__(self, model_name_or_path: str = "ai4bharat/indic-bert", device: Optional[str] = None):
        """
        model_name_or_path:
            - HuggingFace model id (e.g., "ai4bharat/indic-bert")
            - OR local path (e.g., "./models/nlp_tamil")
        """
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = None
        self._model = None
        self.id2label = None
        self.label2id = None
        self._heuristic_fallback = False

    def load(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        name = self.model_name_or_path

        # If local folder exists → use local
        local_path = Path(name)
        if local_path.exists():
            name = str(local_path.resolve())

        try:
            print(f"Loading model from: {name}")

            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = AutoModelForSequenceClassification.from_pretrained(name)

            self._model.to(self.device)
            self._model.eval()

            # Load labels safely
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self.id2label = {int(k): v for k, v in self._model.config.id2label.items()}
                self.label2id = {v: int(k) for k, v in self.id2label.items()}
            else:
                self.id2label = {0: "Genuine Consent", 1: "Neutral", 2: "Coercion"}
                self.label2id = {v: k for k, v in self.id2label.items()}

            print("Model loaded successfully.")

        except Exception as e:
            print("⚠ Model loading failed. Switching to heuristic mode.")
            print("Error:", str(e))

            self._heuristic_fallback = True
            self._tokenizer = None
            self._model = None

            self.id2label = {0: "Genuine Consent", 1: "Neutral", 2: "Coercion"}
            self.label2id = {v: k for k, v in self.id2label.items()}

    def predict(self, text: str) -> Tuple[float, str]:
        if self._model is None and not self._heuristic_fallback:
            self.load()

        # If model loaded properly
        if not self._heuristic_fallback and self._model is not None:
            enc = self._tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors="pt"
            )

            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self._model(**enc)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            pred_id = int(probs.argmax())
            label = self.id2label.get(pred_id, "Neutral")

            coercion_index = self.label2id.get("Coercion", 2)
            coercion_prob = float(probs[coercion_index])

            return coercion_prob, label

        # Heuristic fallback mode
        t = (text or "").lower()

        if re.search(r"\b(zabardasti|pressure|threat|force[d]?)\b", t):
            return 0.9, "Coercion"

        if re.search(r"\b(sondha|virupath[ao]du|consent|willing(ly)?)\b", t):
            return 0.15, "Genuine Consent"

        return 0.5, "Neutral"