# Disposal Guidelines Database
"""
Defined comprehensive disposal guidelines for all categories.
Based on WHO guidelines and Rwanda FDA regulations.
"""

import json

disposal_guidelines = {
    1: {  # Solids
        'category_name': 'Solids (Tablets, Capsules, Powders)',
        'icon': '',
        'quick_summary': {
            'safe_method': 'Encapsulate and dispose in secure landfill or incinerate.',
            'time_required': '30-60 minutes (per batch)',
            'can_diy': True
        },
        'steps': [
            {
                'step': 1,
                'title': 'Preparation and Sorting',
                'actions': [
                    'Wear appropriate Personal Protective Equipment (PPE): gloves, mask, and protective gown',
                    'Sort expired or unused tablets separately from regular waste',
                    'Check expiry dates and verify medicine names',
                    'Document the quantity and type of medicine being disposed',
                    'Keep a disposal log with date, medicine name, quantity, and reason for disposal'
                ]
            },
            {
                'step': 2,
                'title': 'Packaging Removal',
                'actions': [
                    'Remove medicines from outer cardboard or paper packaging',
                    'Keep medicines in their original blister packs or inner packaging',
                    'Do NOT remove individual tablets from blister packs',
                    'Separate packaging materials for recycling if possible',
                    'Ensure labels remain legible for identification'
                ]
            },
            {
                'step': 3,
                'title': 'Container Preparation',
                'actions': [
                    'Use clean plastic drums (HDPE) or steel drums with secure lids',
                    'Label drums clearly: "PHARMACEUTICAL WASTE - SOLIDS"',
                    'Add date of collection on the label',
                    'If handling large quantities of one drug, mix with other medicines',
                    'Fill containers to maximum 75% capacity for safe handling'
                ]
            },
            {
                'step': 4,
                'title': 'Disposal Method Selection',
                'options': [
                    'Encapsulation: Mix with cement/lime in 55-gallon drums (preferred for small-medium quantities)',
                    'Inertization: Mix with water, cement, lime, and sawdust (for medium quantities)',
                    'High-temperature incineration: >1000°C (for large quantities or high-risk drugs)',
                    'Secure landfill: Only for properly encapsulated waste in designated areas'
                ]
            },
            {
                'step': 5,
                'title': 'Transport and Documentation',
                'actions': [
                    'Seal drums securely before transport',
                    'Complete waste transfer documentation',
                    'Use authorized transporters with permits',
                    'Transport to Rwanda FDA-authorized disposal facility',
                    'Keep records for minimum 3 years',
                    'Obtain disposal certificate from facility'
                ]
            }
        ],
        'prohibitions': [
            'Do NOT flush tablets down toilet or sink (unless specifically listed in FDA flush list)',
            'Do NOT burn in open air or low-temperature incinerators',
            'Do NOT dispose in regular household waste without encapsulation',
            'Do NOT mix with infectious or sharp waste',
            'Do NOT crush or pulverize high-risk drugs without proper containment',
            'Do NOT dispose in water bodies or drainage systems'
        ],
        'safety': [
            'Always wear appropriate PPE (gloves, mask, goggles)',
            'Work in well-ventilated areas',
            'Wash hands thoroughly with soap after handling',
            'Keep medicines away from children during disposal process',
            'Have spill kit readily available',
            'Clean work surfaces with disinfectant after disposal activities',
            'Report any accidental exposure immediately'
        ]
    },
    2: {  # Liquids
        'category_name': 'Liquids (Solutions, Injections, Syrups)',
        'icon': '💧',
        'quick_summary': {
            'safe_method': 'Dilute biodegradable liquids for sewer disposal (with treatment); incinerate non-biodegradable.',
            'time_required': '15-45 minutes (per batch)',
            'can_diy': True
        },
        'steps': [
            {
                'step': 1,
                'title': 'Initial Assessment and Classification',
                'actions': [
                    'Identify if liquid is biodegradable or non-biodegradable',
                    'Check if sewage treatment plant is available and authorized',
                    'Verify medicine risk level (HIGH/MEDIUM/LOW)',
                    'Wear appropriate PPE: chemical-resistant gloves, goggles, face shield, gown',
                    'Document quantity in liters or milliliters',
                    'Check container integrity for leaks or damage'
                ]
            },
            {
                'step': 2,
                'title': 'Segregation by Risk Level',
                'actions': [
                    'BIODEGRADABLE: vitamins, glucose, saline, amino acids (segregate for dilution disposal)',
                    'NON-BIODEGRADABLE: antibiotics, antineoplastics (segregate for incineration)',
                    'HIGH-RISK: antineoplastics, cytotoxic drugs (keep completely separate)',
                    'Check for compatibility issues before mixing',
                    'Use clearly labeled containers for each category'
                ]
            },
            {
                'step': 3,
                'title': 'Disposal Method Selection',
                'options': [
                    'Sewer disposal: ONLY for biodegradable liquids with sewage treatment (dilute 1:10 minimum)',
                    'Pit disposal: Dig pit 1.5m deep, line with clay, dispose and cover with soil (if no sewer)',
                    'Chemical treatment: Neutralize acids/bases, precipitate heavy metals before disposal',
                    'High-temperature incineration: >1200°C for non-biodegradable and high-risk liquids',
                    'Return to supplier: For large quantities of controlled substances'
                ]
            },
            {
                'step': 4,
                'title': 'Safe Disposal Execution',
                'actions': [
                    'For sewer disposal: Dilute with at least 10 parts water',
                    'Pour slowly to avoid splashing and aerosol formation',
                    'Flush with additional water after disposal',
                    'Rinse empty containers three times with water',
                    'Crush containers after rinsing to prevent reuse',
                    'Ensure adequate ventilation during entire process'
                ]
            },
            {
                'step': 5,
                'title': 'Documentation and Monitoring',
                'actions': [
                    'Record type, quantity, and disposal method used',
                    'Document date, time, and personnel involved',
                    'Obtain disposal certificates for incinerated waste',
                    'Monitor disposal site for environmental impact',
                    'Keep records for minimum 3 years',
                    'Report any spills or accidents immediately'
                ]
            }
        ],
        'prohibitions': [
            'NEVER dispose antineoplastic drugs in sewer systems',
            'NEVER dispose antibiotics in sewer (contributes to antimicrobial resistance)',
            'Do NOT pour concentrated medicines directly into sewers',
            'Do NOT reuse medicine containers for other purposes',
            'Do NOT mix incompatible chemicals (acids with bases, oxidizers with reducers)',
            'Do NOT dispose in storm drains or natural water bodies',
            'Do NOT incinerate in low-temperature incinerators (<800°C)'
        ],
        'safety': [
            'Prevent splashing during dilution and disposal',
            'Ensure adequate ventilation (outdoor or fume hood)',
            'Have chemical spill kit readily available',
            'Clean spills immediately with absorbent material',
            'Neutralize spills if possible before cleanup',
            'Dispose of cleanup materials as hazardous waste',
            'Provide eyewash station and safety shower nearby',
            'Never work alone when handling high-risk liquids'
        ]
    },
    3: {  # Semisolids
        'category_name': 'Semisolids (Creams, Ointments, Gels)',
        'icon': '🧴',
        'quick_summary': {
            'safe_method': 'Contain and dispose via incineration or secure landfill.',
            'time_required': '10-20 minutes',
            'can_diy': True
        },
        'steps': [
            {
                'step': 1,
                'title': 'Preparation',
                'actions': [
                    'Wear PPE: gloves and protective gown',
                    'Collect expired/unused semisolid medicines',
                    'Keep in original containers where possible'
                ]
            },
            {
                'step': 2,
                'title': 'Containment',
                'actions': [
                    'Place in sealed plastic bags or containers',
                    'Label as "PHARMACEUTICAL WASTE - SEMISOLIDS"',
                    'Do not empty tubes completely (to maintain identification)'
                ]
            },
             {
                'step': 3,
                'title': 'Disposal',
                'options': [
                    'Incineration: High-temperature (>1000°C) preferred',
                    'Landfill: Only in secure pharmaceutical waste section',
                    'Encapsulation: Mix with cement for small quantities'
                ]
            }
        ],
        'prohibitions': [
            'Do NOT flush down toilet or sink',
            'Do NOT dispose in regular trash',
            'Do NOT burn in open air'
        ],
        'safety': [
            'Wear gloves to prevent skin contact',
            'Wash hands after handling',
            'Avoid contact with eyes'
        ]
    },
    4: {  # Aerosols
        'category_name': 'Aerosols and Inhalers',
        'icon': '💨',
        'quick_summary': {
            'safe_method': 'Return to pharmacy or specialized hazardous waste facility.',
            'time_required': 'Depends on facility access',
            'can_diy': False
        },
        'steps': [
            {
                'step': 1,
                'title': 'Special Handling',
                'actions': [
                    'Do NOT puncture or incinerate pressurized containers',
                    'Keep in original packaging',
                    'Ensure containers are not damaged'
                ]
            },
            {
                'step': 2,
                'title': 'Depressurization',
                'actions': [
                    'Release pressure in controlled manner if possible',
                    'Work in well-ventilated area',
                    'Follow manufacturer instructions for disposal'
                ]
            },
            {
                'step': 3,
                'title': 'Disposal',
                'options': [
                    'Return to pharmacy or manufacturer program',
                    'Specialized waste facility for pressurized containers',
                    'Hazardous waste collection program'
                ]
            }
        ],
        'prohibitions': [
            'NEVER puncture pressurized containers',
            'Do NOT incinerate',
            'Do NOT expose to heat or flames',
            'Do NOT crush'
        ],
        'safety': [
            'Handle with extreme care',
            'Store in cool, dry place until disposal',
            'Keep away from heat sources',
            'Use safety goggles when handling damaged containers'
        ]
    },
    5: {  # Biological
        'category_name': 'Biological Waste (Vaccines, Blood Products)',
        'icon': '💉',
        'quick_summary': {
            'safe_method': 'Dispose via high-temperature incineration at specialized facility.',
            'time_required': 'Depends on facility access',
            'can_diy': False
        },
        'steps': [
            {
                'step': 1,
                'title': 'Biohazard Protocol',
                'actions': [
                    'Treat as potentially infectious material',
                    'Wear full PPE including face shield',
                    'Use puncture-resistant containers',
                    'Label with biohazard symbol'
                ]
            },
            {
                'step': 2,
                'title': 'Containment',
                'actions': [
                    'Place in red biohazard bags',
                    'Seal containers securely',
                    'Store in designated biohazard area',
                    'Maintain cold chain if required'
                ]
            },
            {
                'step': 3,
                'title': 'Disposal',
                'options': [
                    'High-temperature incineration: >1200°C (required)',
                    'Autoclave before disposal (if applicable)',
                    'Specialized biomedical waste facility'
                ]
            }
        ]
    }
}

def display_disposal_guide(category_id, medicine_name="Medicine"):
    """
    Displays the detailed disposal guidelines for a given category.

    Args:
        category_id (int): The ID of the disposal category (1-5).
        medicine_name (str): The name of the medicine for context.
    """
    guidelines = disposal_guidelines.get(category_id)

    if not guidelines:
        print(f"\n No detailed guidelines found for Category {category_id}.")
        return

    print("\n" + "="*80)
    print(f"DETAILED DISPOSAL GUIDELINES FOR {medicine_name.upper()}")
    print("="*80)
    print(f"\nCategory: {guidelines['category_name']}")

    if 'steps' in guidelines:
        print("\n" + "="*80)
        print("STEP-BY-STEP DISPOSAL PROCEDURE")
        print("="*80)

        for step in guidelines['steps']:
            print(f"\nSTEP {step['step']}: {step['title']}")
            print("-" * 80)

            if 'actions' in step:
                for action in step['actions']:
                    print(f"  - {action}")

            if 'options' in step:
                print("  Options:")
                for option in step['options']:
                    print(f"    * {option}")

    # Prohibitions
    if 'prohibitions' in guidelines:
        print("\n" + "="*80)
        print("PROHIBITIONS")
        print("="*80)
        for prohibition in guidelines['prohibitions']:
            print(f"  X {prohibition}")

    # Safety
    if 'safety' in guidelines:
        print("\n" + "="*80)
        print("SAFETY PRECAUTIONS")
        print("="*80)
        for safety in guidelines['safety']:
            print(f"  ! {safety}")

    print("\n" + "="*80)


def quick_reference_card():
    """Displays a quick reference card for disposal categories."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "QUICK REFERENCE CARD" + " "*28 + "║")
    print("╚" + "="*68 + "╝\n")

    print(" DISPOSAL CATEGORIES:")
    print("-" * 70)

    for cat_id, guidelines in disposal_guidelines.items():
        print(f" {cat_id} {guidelines['icon']} {guidelines['category_name']}")
        if 'quick_summary' in guidelines:
            summary = guidelines['quick_summary']
            print(f"   Recommended Method: {summary['safe_method']}")
            can_diy = "Yes" if summary['can_diy'] else "Requires special disposal"
            print(f"   Can Do At Home: {can_diy}")
        print("-" * 70)

    print("\n For detailed guidelines, use the predictor or refer to the full documentation.")
    print("="*70 + "\n")


print("Disposal guidelines database created successfully.")
print(f"Total categories: {len(disposal_guidelines)}")
for cat_id, guidelines in disposal_guidelines.items():
    print(f"  Category {cat_id}: {guidelines['category_name']}")
