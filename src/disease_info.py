"""Static, general information shown next to a prediction.

Deliberately descriptive only: what the disease looks like and how it is
generally spread.  **No** treatment product, dosage or calendar advice is
included, because a classifier on leaf photographs cannot know the agronomic
situation (growth stage, hybrid resistance, weather, local regulations) and the
project must not present itself as professional advice.
"""

from __future__ import annotations

#: canonical class key -> display name -> information block
DISEASE_INFORMATION: dict[str, dict[str, str]] = {
    "healthy": {
        "display_name": "Healthy",
        "summary": "No visible leaf disease in this image.",
        "description": (
            "The leaf shows green lamina without the pustules, lesions or "
            "elongated blotches that the model associates with the three "
            "disease classes. Absence of visible symptoms in one photograph "
            "does not prove the plant is disease-free: symptoms can be "
            "localized elsewhere on the plant or not yet visible."
        ),
        "signs": "Even green colouration, no powdery pustules, no cigar-shaped lesions.",
    },
    "common_rust": {
        "display_name": "Common Rust",
        "summary": "Fungal disease (Puccinia sorghi) forming rust-coloured pustules.",
        "description": (
            "A fungal disease characterised by small, scattered, rust-coloured "
            "pustules (uredinia) that rupture the leaf surface and release "
            "powdery reddish-brown spores. They appear on both leaf surfaces. "
            "Severity generally increases with cool, humid weather and with "
            "long periods of leaf wetness, and heavy infection can shorten "
            "grain filling."
        ),
        "signs": "Cinnamon-brown to reddish powdery pustules, often on both sides of the leaf.",
    },
    "northern_leaf_blight": {
        "display_name": "Northern Leaf Blight",
        "summary": "Fungal disease (Exserohilum turcicum) with large cigar-shaped lesions.",
        "description": (
            "A fungal disease recognised by long, elliptical to cigar-shaped "
            "grey-green lesions that later turn tan and may develop dark "
            "olive sporulation. Lesions usually start on lower leaves and move "
            "up the plant; extensive early blighting can reduce leaf area "
            "substantially. Spread is favoured by prolonged humid conditions "
            "and cooler temperatures, and by residue-borne inoculum."
        ),
        "signs": "Large spindle/cigar-shaped lesions, typically 3-15 cm, often with parallel margins.",
    },
    "gray_leaf_spot": {
        "display_name": "Gray Leaf Spot",
        "summary": "Fungal disease (Cercospora zeae-maydis) with rectangular, vein-bounded lesions.",
        "description": (
            "A fungal disease, also referred to as Cercospora leaf spot, "
            "producing narrow, rectangular, tan-to-grey lesions whose edges "
            "run parallel to the leaf veins, giving a characteristic "
            "block-like shape. Lesions typically appear first on lower leaves "
            "and can coalesce under warm, humid weather, reducing the "
            "photosynthetic area of the canopy."
        ),
        "signs": "Rectangular grey/tan lesions bounded by leaf veins; box-like rather than spindle-like.",
    },
}

DISCLAIMER = (
    "This system provides an AI-based image classification result and should "
    "not replace diagnosis or advice from a qualified agricultural professional."
)

LOW_CONFIDENCE_MESSAGE = (
    "Low-confidence prediction. Consider providing a clearer, close-up image of "
    "a single leaf on a plain background, and consult an agricultural extension "
    "officer or plant clinic before making any management decision."
)

#: Model-level limitation shown in the UI so it is impossible to miss.
SCOPE_WARNING = (
    "The model was trained on controlled, single-leaf photographs from the "
    "PlantVillage dataset (plain background, expert-labelled). Field photographs "
    "with mixed symptoms, other diseases, nutrient deficiency, pest damage or "
    "unusual lighting are outside that setting, so real-world accuracy is "
    "expected to be lower than the benchmark figure reported for the test split."
)


def information_for(class_name: str) -> dict[str, str]:
    """Look up the information block by display name or canonical key."""
    key = class_name.strip().lower().replace(" ", "_")
    if key in DISEASE_INFORMATION:
        return DISEASE_INFORMATION[key]
    for info in DISEASE_INFORMATION.values():
        if info["display_name"].lower() == class_name.strip().lower():
            return info
    return {
        "display_name": class_name,
        "summary": "No information entry for this label.",
        "description": "This project only describes the four maize classes it was trained on.",
        "signs": "",
    }
