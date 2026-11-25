"""The cause taxonomy. The LLM labels comments against these descriptions,
so adding a cause is one new entry here and no new code."""

THEMES = {
    "fit": {
        "name": "Size and fit",
        "about": "Size, fit or sizing guidance was wrong: too small, too big, straps too short, falls out.",
        "fix": "Add a size chart with body measurements and say whether it runs small or large.",
    },
    "desc": {
        "name": "Not as described",
        "about": "Photos, specs or claims on the page do not match the product received.",
        "fix": "Match photos and claims to the real product and remove claims customers contradict.",
    },
    "dur": {
        "name": "Breaks or wears out",
        "about": "Product broke, tore, stopped working or wore out within normal use.",
        "fix": "Flag the weak component with the supplier and add care or lifetime details.",
    },
    "perf": {
        "name": "Performance gap",
        "about": "Works, but not as well as promised: battery hours, noise cancelling, waterproofing.",
        "fix": "Replace headline numbers with tested ranges and show the test conditions.",
    },
    "dmg": {
        "name": "Damaged in transit",
        "about": "Arrived damaged: crushed box, dents, scuffs, torn packaging.",
        "fix": "Review packaging for this item and reinforce where damage reports cluster.",
    },
    "miss": {
        "name": "Missing parts or instructions",
        "about": "Parts, accessories or instructions were missing or unclear.",
        "fix": "Print the contents list on the page and add a quick-start card to the box.",
    },
    "ship": {
        "name": "Slow delivery",
        "about": "Delivery took longer than the customer expected.",
        "fix": "Show a realistic delivery window so the promise matches what arrives.",
    },
}
