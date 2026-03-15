def explain_fault(fault):

    explanations = {
        "engine_knocking": "Engine knocking occurs when combustion happens unevenly in the cylinder. Possible causes include worn spark plugs or incorrect ignition timing.",
        "loose_belt": "A loose belt may produce squealing sounds due to insufficient tension.",
        "bearing_noise": "Bearing noise often indicates worn or poorly lubricated bearings.",
        "normal_engine": "The engine sound appears normal."
    }

    return explanations.get(fault, "Unknown engine fault detected.")
