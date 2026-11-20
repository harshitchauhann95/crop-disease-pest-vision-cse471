"""
src/advisory.py
---------------
Agronomic Advisory & Management Engine for SIH26131:
Translates diagnostic predictions (disease, pest, severity stage) into actionable,
field-tested agricultural management guidelines adhering to ICAR / CIBRC standards.
Encourages sustainable IPM (Integrated Pest Management) and local KVK verification.
"""

from typing import Dict, Any, Optional, List


# Comprehensive Agronomic Knowledge Base
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Tomato___Early_Blight": {
        "crop": "Tomato",
        "common_name": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "favorable_conditions": "Warm temperatures (24-29°C) accompanied by heavy dew, frequent rain, or overhead irrigation.",
        "symptoms": "Concentric 'target-board' dark brown rings on older lower leaves, often surrounded by a chlorotic yellow halo.",
        "cultural_practices": [
            "Prune and safely burn or bury severely infected lower foliage to prevent spore splash.",
            "Switch from overhead sprinkler irrigation to drip irrigation to keep foliage dry.",
            "Practice 3-year crop rotation with non-solanaceous crops (e.g., corn, legumes).",
            "Apply straw mulch around the base of plants to create a physical barrier against soil-borne spores."
        ],
        "biological_controls": [
            "Foliar spray of Trichoderma harzianum or Bacillus subtilis (5-10 g/L).",
            "Apply cold-pressed Neem seed oil (1500 ppm) at 3-5 mL/L with mild surfactant as early preventive measure."
        ],
        "chemical_interventions": {
            "mild": "Preventive contact spray: Mancozeb 75% WP @ 2.0-2.5 g/L or Copper Oxychloride 50% WP @ 2.5 g/L.",
            "moderate": "Azoxystrobin 23% SC @ 1.0 mL/L or Chlorothalonil 75% WP @ 2.0 g/L.",
            "severe": "Systemic curative spray: Difenoconazole 25% EC @ 0.5-1.0 mL/L or Tebuconazole 25.9% EC @ 1.0 mL/L."
        },
        "scouting_interval": "Scout every 3-4 days under overcast, humid conditions."
    },
    "Tomato___Late_Blight": {
        "crop": "Tomato",
        "common_name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "favorable_conditions": "Cool (15-20°C), highly humid weather (>90% RH) with continuous cloud cover.",
        "symptoms": "Large, irregular water-soaked dark grayish lesions on leaves and stems; white cottony fungal growth on leaf undersides in high humidity.",
        "cultural_practices": [
            "Immediately rogue out and destroy severely blighted plants to curb rapid community transmission.",
            "Ensure wide plant spacing and staking to maximize airflow and canopy sunlight penetration.",
            "Eliminate volunteer tomato and potato plants in adjacent plots."
        ],
        "biological_controls": [
            "Bio-formulation sprays of Pseudomonas fluorescens (10 g/L) applied prophylactically."
        ],
        "chemical_interventions": {
            "mild": "Prophylactic spray: Mancozeb 75% WP @ 2.5 g/L or Propineb 70% WP @ 2.0 g/L.",
            "moderate": "Metalaxyl 8% + Mancozeb 64% WP @ 2.5 g/L.",
            "severe": "Cymoxanil 8% + Mancozeb 64% WP @ 2.0 g/L or Dimethomorph 50% WP @ 1.0 g/L."
        },
        "scouting_interval": "Daily scouting during persistent cool, damp weather."
    },
    "Tomato___Healthy": {
        "crop": "Tomato",
        "common_name": "Healthy Foliage",
        "pathogen": "None detected",
        "favorable_conditions": "Optimal balanced vegetative growth.",
        "symptoms": "Vigorous green leaves, turgid foliage, devoid of chlorosis, necrosis, or pest lesions.",
        "cultural_practices": [
            "Maintain balanced N-P-K fertilization; avoid excessive nitrogen that attracts aphids and causes lush susceptible foliage.",
            "Maintain consistent drip moisture schedule."
        ],
        "biological_controls": ["Routine prophylactic neem cake application in soil at root zone."],
        "chemical_interventions": {"mild": "No synthetic chemical intervention required.", "moderate": "None", "severe": "None"},
        "scouting_interval": "Weekly routine field scouting."
    },
    "Potato___Early_Blight": {
        "crop": "Potato",
        "common_name": "Early Blight",
        "pathogen": "Alternaria solani",
        "favorable_conditions": "Warm days, humid nights, stressed crop canopy.",
        "symptoms": "Dark brown circular spots with characteristic concentric rings on mature leaflets.",
        "cultural_practices": [
            "Ensure adequate potassium and nitrogen nutrition to avoid premature canopy senescence.",
            "Avoid sprinkler irrigation in the late evening."
        ],
        "biological_controls": ["Trichoderma viride foliar wash @ 5 g/L."],
        "chemical_interventions": {
            "mild": "Mancozeb 75 WP @ 2.5 g/L.",
            "moderate": "Metiram 70% WG @ 2.0 g/L.",
            "severe": "Pyraclostrobin 20% WG @ 1.0 g/L."
        },
        "scouting_interval": "Every 4-5 days."
    },
    "Rice___Leaf_Blast": {
        "crop": "Rice",
        "common_name": "Leaf Blast",
        "pathogen": "Magnaporthe oryzae (Pyricularia oryzae)",
        "favorable_conditions": "High relative humidity (>90%), night temperatures 20-24°C, high nitrogen fertilizer doses.",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with gray or whitish centers and brown necrotic borders.",
        "cultural_practices": [
            "Avoid excessive split doses of urea; balance with muriate of potash (MOP).",
            "Maintain shallow water standing (2-5 cm) to avoid crop water stress."
        ],
        "biological_controls": ["Seed and seedling dip with Pseudomonas fluorescens @ 10 g/kg seed."],
        "chemical_interventions": {
            "mild": "Tricyclazole 75% WP @ 0.6 g/L preventive spray.",
            "moderate": "Isoprothiolane 40% EC @ 1.5 mL/L.",
            "severe": "Kasugamycin 3% SL @ 2.0 mL/L or Tricyclazole + Tebuconazole @ 1.0 g/L."
        },
        "scouting_interval": "Every 3 days during tillering and panicle initiation."
    },
    "Apple___Apple_Scab": {
        "crop": "Apple",
        "common_name": "Apple Scab",
        "pathogen": "Venturia inaequalis",
        "favorable_conditions": "Cool, wet spring weather during bud burst and petal fall.",
        "symptoms": "Olive-green, velvety spots on upper leaf surfaces that turn corky and black.",
        "cultural_practices": [
            "Shred and rake fallen autumn orchard leaves or apply 5% urea spray post-harvest to speed decomposition.",
            "Prune dense canopy trees to accelerate leaf drying."
        ],
        "biological_controls": ["Serenade (Bacillus amyloliquefaciens) during early green tip stage."],
        "chemical_interventions": {
            "mild": "Captan 50% WP @ 2.5 g/L or Dodine 65% WP @ 1.0 g/L.",
            "moderate": "Myclobutanil 10% WP @ 0.8 g/L.",
            "severe": "Kresoxim-methyl 44.3% SC @ 0.5 mL/L."
        },
        "scouting_interval": "Twice weekly from silver tip to petal fall."
    }
}

PEST_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "aphid": {
        "pest_name": "Aphids (Green peach / Melon aphid)",
        "damage_mechanism": "Sucking sap from succulent shoot tips and abaxial leaf surfaces, excreting sticky honeydew attracting sooty mold.",
        "mechanical_cultural": [
            "Install bright yellow sticky cards (15-20 traps per acre) just above crop canopy level.",
            "Dislodge clustered nymphs using a targeted strong jet of water."
        ],
        "biological_control": "Release predatory ladybird beetles (Coccinella septempunctata) or green lacewings (Chrysoperla carnea).",
        "chemical_guideline": "Neem oil 1500 ppm @ 3 mL/L or Flonicamid 50% WG @ 0.3 g/L or Imidacloprid 17.8% SL @ 0.3 mL/L."
    },
    "armyworm": {
        "pest_name": "Fall Armyworm / Spodoptera litura",
        "damage_mechanism": "Voracious chewing larvae skeletonizing leaves and boring into whorls leaving characteristic frass.",
        "mechanical_cultural": [
            "Hand-pick and crush egg masses and early instars in small-scale holdings.",
            "Install pheromone traps (5 per acre) for monitoring adult moth emergence."
        ],
        "biological_control": "Spray Bacillus thuringiensis (Bt) kurstaki @ 2 g/L or NPV (Nuclear Polyhedrosis Virus) @ 250 LE/ha.",
        "chemical_guideline": "Emamectin benzoate 5% SG @ 0.4 g/L or Chlorantraniliprole 18.5% SC @ 0.3 mL/L."
    },
    "whitefly": {
        "pest_name": "Silverleaf Whitefly (Bemisia tabaci)",
        "damage_mechanism": "Sap depletion causing chlorotic spotting; key vector for devastating Geminiviruses (e.g., Tomato Leaf Curl).",
        "mechanical_cultural": [
            "Set up yellow sticky traps around greenhouse perimeter and field borders.",
            "Use reflective silver plastic mulches to deter adult landing."
        ],
        "biological_control": "Entomopathogenic fungus Verticillium lecanii or Paecilomyces fumosoroseus @ 5 g/L.",
        "chemical_guideline": "Diafenthiuron 50% WP @ 1.2 g/L or Spiromesifen 22.9% SC @ 1.0 mL/L."
    },
    "leaf_miner": {
        "pest_name": "Serpentine Leaf Miner (Liriomyza spp.)",
        "damage_mechanism": "Larvae mine between upper and lower leaf epidermises, creating winding white serpentine trails.",
        "mechanical_cultural": [
            "Pinch and destroy mined leaves displaying active larvae.",
            "Use yellow sticky traps to catch flying adult flies."
        ],
        "biological_control": "Encourage native parasitic wasps (Diglyphus isaea).",
        "chemical_guideline": "Abamectin 1.9% EC @ 0.5 mL/L or Cyromazine 75% WP @ 0.25 g/L."
    }
}


def generate_agronomic_advisory(
    diagnosis: str,
    confidence: float,
    severity_pct: float = 0.0,
    severity_stage: str = "Mild",
    detected_pests: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Synthesizes diagnostic prediction, severity percentage, and detected pests
    into a comprehensive, structured advisory report.
    """
    disease_info = DISEASE_KNOWLEDGE_BASE.get(diagnosis, None)

    if disease_info is None:
        # Fallback generic advice
        disease_info = {
            "crop": "Crop Foliage",
            "common_name": diagnosis.replace("___", " - ").replace("_", " "),
            "pathogen": "Foliar pathogen",
            "favorable_conditions": "Humid, moist canopy microclimate.",
            "symptoms": "Localized foliar lesion spots or chlorosis.",
            "cultural_practices": [
                "Remove and safely dispose of affected plant foliage.",
                "Ensure adequate aeration and avoid wetting leaves when irrigating."
            ],
            "biological_controls": ["Prophylactic cold-pressed neem oil spray (3-5 mL/L)."],
            "chemical_interventions": {
                "mild": "Broad-spectrum copper oxychloride 50 WP @ 2.5 g/L.",
                "moderate": "Mancozeb 75 WP @ 2.0 g/L.",
                "severe": "Consult local Krishi Vigyan Kendra (KVK) for systemic fungicide."
            },
            "scouting_interval": "Scout twice weekly."
        }

    # Select severity tier
    if severity_pct < 10.0:
        tier = "mild"
        severity_label = "Stage 1: Low / Incipient Infestation"
        action_urgency = "Prophylactic & Cultural"
    elif severity_pct <= 30.0:
        tier = "moderate"
        severity_label = "Stage 2: Moderate Severity"
        action_urgency = "Targeted Intervention Required"
    else:
        tier = "severe"
        severity_label = "Stage 3: High Severity / Advanced Spread"
        action_urgency = "Immediate Containment Essential"

    recommended_chemical = disease_info["chemical_interventions"].get(tier, disease_info["chemical_interventions"].get("mild"))

    pest_advisories = []
    if detected_pests:
        for p in detected_pests:
            p_clean = p.lower().strip()
            if p_clean in PEST_KNOWLEDGE_BASE:
                pest_advisories.append(PEST_KNOWLEDGE_BASE[p_clean])

    advisory_report = {
        "crop": disease_info["crop"],
        "diagnosis": disease_info["common_name"],
        "pathogen": disease_info["pathogen"],
        "model_confidence": f"{confidence * 100:.1f}%",
        "affected_leaf_area": f"{severity_pct:.1f}%",
        "severity_level": severity_label,
        "action_urgency": action_urgency,
        "favorable_conditions": disease_info["favorable_conditions"],
        "symptom_confirmation": disease_info["symptoms"],
        "cultural_measures": disease_info["cultural_practices"],
        "biological_controls": disease_info["biological_controls"],
        "targeted_intervention": recommended_chemical,
        "scouting_schedule": disease_info["scouting_interval"],
        "pest_guidance": pest_advisories,
        "regulatory_disclaimer": (
            "NOTICE: This automated diagnostic advisory is generated by deep learning models for decision-support. "
            "Farmers are strongly advised to confirm symptoms with their local Krishi Vigyan Kendra (KVK), "
            "District Agricultural Extension Officer, or State Agricultural University scientist before procuring "
            "or applying any commercial chemical fungicides or pesticides."
        )
    }

    return advisory_report
