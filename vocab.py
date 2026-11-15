"""House drum vocabulary: every way we might describe a one-shot to the teacher.

Compiled analytically (producer and sample-pack language, drum machine history, house subgenres,
sound anatomy). `vocab_screen.py` measures what each entry actually does to the sound and writes
`vocab_catalogue.csv`; the generator then uses only entries that proved themselves.

Entry = "text" (applies to every category) or ("text", "kick,snare") (only those categories).
Axes marked PRE go before the instrument name ("TR-909 kick drum"); the rest are appended
after it ("kick drum, punchy").
"""

PRE = {"source", "genre"}

VOCAB = {
    # where the sound comes from: machines, samplers, synthesis, acoustic
    "source": [
        "TR-909", "TR-808", "TR-707", "TR-606", "TR-727", "CR-78", "LinnDrum", "Oberheim DMX",
        "Simmons electronic", "SP-1200", "MPC60", "drum machine", "vintage drum machine",
        "analog drum machine", "digital drum machine", "acoustic", "live studio", "electronic",
        "synthesized", "FM synthesized", "modular synth", "sampled", "breakbeat",
    ],
    # scene and era
    "genre": [
        "house", "deep house", "tech house", "acid house", "Chicago house", "classic Chicago house",
        "Detroit techno", "New York garage house", "UK garage", "French house", "disco", "Italo disco",
        "minimal techno", "microhouse", "lo-fi house", "progressive house", "afro house",
        "tribal house", "soulful house", "jackin house", "electro", "techno", "hip-hop",
        "80s", "90s", "modern", "old-school", "warehouse rave",
    ],
    # what was done to it afterwards
    "processing": [
        "12-bit sampler crunch", "bitcrushed", "tape saturated", "overdriven", "distorted",
        "heavily compressed", "limited", "gated", "low-pass filtered", "high-pass filtered",
        "sampled from vinyl", "resampled", "layered", "pitched down", "pitched up", "detuned",
        "transient shaped", "saturated", "lo-fi", "hi-fi", "analog warmth", "digital",
    ],
    # anatomy 1: the first few milliseconds
    "transient": [
        "hard click transient", "soft attack", "sharp attack", "fast attack", "slow attack",
        "snappy", "clicky", "punchy", "slappy", "spiky", "rounded attack", "no click",
        ("beater click", "kick"), ("felt beater", "kick"),
    ],
    # anatomy 2: the middle
    "body": [
        "thick body", "thin body", "hollow", "boxy", "round", "full", "fat", "tubby", "woody",
        "knocking mid punch", "resonant", "tonal", "ringing",
        ("sub-heavy", "kick"), ("deep sub body", "kick"), ("fast pitch drop", "kick"),
        ("slow pitch sweep", "kick"),
    ],
    # anatomy 3: how it ends
    "tail": [
        "short", "very short", "long", "tight", "fast decay", "slow decay", "long tail",
        "gated tail", "ringing tail", "noisy tail", "damped", "muted", "choked", "sustained", "boomy",
    ],
    # overall tone
    "tone": [
        "bright", "dark", "dull", "crisp", "airy", "sizzling", "shimmering", "hissy", "glassy",
        "metallic", "tinny", "harsh", "smooth", "mellow", "muffled", "open", "nasal", "midrangey",
        "bassy", "warm", "cold",
    ],
    "size": [
        "big", "huge", "massive", "small", "tiny", "heavy", "light", "powerful", "weak",
        "hard-hitting", "soft", "gentle", "aggressive", "subtle",
    ],
    "texture": [
        "clean", "pristine", "polished", "raw", "gritty", "crunchy", "grainy", "dirty", "trashy",
        "dusty", "rough", "silky", "fuzzy", "crackly", "noisy",
    ],
    "pitch": ["low-pitched", "mid-pitched", "high-pitched", "tuned low", "tuned high", "deep"],
    # the sound spelled out
    "onomatopoeia": [
        ("thud", "kick"), ("thump", "kick"), ("boom", "kick"), ("knock", "kick,snare"),
        ("tick", "hat_closed,hat_open"), ("tss", "hat_closed,hat_open"), ("chick", "hat_closed,hat_open"),
        ("snap", "clap,snare"), ("crack", "clap,snare"), ("smack", "clap,snare"), ("slap", "clap,snare"),
        ("pop", "kick,clap,snare"), ("click", "kick,hat_closed"), ("ping", "hat_closed,hat_open"),
        ("whack", "snare,clap"),
    ],
    # how and on what it is played, per instrument
    "performance": [
        "hit hard", "hit softly",
        ("single hand clap", "clap"), ("group of people clapping once", "clap"), ("layered flam", "clap"),
        ("wide stereo layers", "clap"), ("clap with snare layer", "clap"), ("finger snap layer", "clap"),
        ("rimshot", "snare"), ("snare wires rattle", "snare"), ("piccolo snare", "snare"),
        ("fat low snare", "snare"), ("snare with clap layer", "snare"), ("brushed", "snare"),
        ("pedal hi-hat", "hat_closed"), ("tightly closed", "hat_closed"), ("stick tip", "hat_closed,hat_open"),
        ("stick edge", "hat_closed,hat_open"), ("half-open", "hat_open"), ("loose", "hat_open"),
        ("small 13 inch cymbals", "hat_closed,hat_open"), ("large 15 inch cymbals", "hat_closed,hat_open"),
        ("22 inch bass drum", "kick"), ("small 18 inch bass drum", "kick"), ("layered with sub", "kick"),
        ("layered with top click", "kick"),
    ],
    # filler words that should do nothing: they set the "any extra word" noise floor
    "control": ["nice", "good", "sound", "sample", "audio"],
}

# the instrument names the screening uses as its neutral base
BASE = {"kick": "kick drum", "clap": "clap", "snare": "snare drum", "hat_closed": "closed hi-hat",
        "hat_open": "open hi-hat"}


def entries():
    """Yield (axis, text, categories) for every vocabulary entry."""
    for axis, words in VOCAB.items():
        for w in words:
            text, cats = (w, list(BASE)) if isinstance(w, str) else (w[0], w[1].split(","))
            yield axis, text, cats


if __name__ == "__main__":
    from collections import Counter
    c = Counter(a for a, _, _ in entries())
    print(sum(c.values()), "entries:", dict(c))
