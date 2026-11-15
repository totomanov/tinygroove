#!/usr/bin/env bash
# Render 12 test one-shots with the Stable Audio 3 MLX runtime.
#
# Usage:
#   extras/sa3_samples.sh                 # small-sfx, seed 42
#   MODEL=medium extras/sa3_samples.sh    # A/B against medium
#   SEED=7 STEPS=8 CFG=1.0 extras/sa3_samples.sh
#   ONLY=kick,clap extras/sa3_samples.sh  # re-render a subset while tuning
#   SET=vars SEEDS="1 2" extras/sa3_samples.sh  # kick/clap/hat variations, 2 seeds each
#
# Output: renders/<run>/<name>.wav plus renders/<run>/meta.jsonl (prompt/seed/params per file).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SA3_DIR="${SA3_DIR:-$HERE/stable-audio-3/optimized/mlx}"
MODEL="${MODEL:-sm-sfx}"           # sm-sfx | medium
SEED="${SEED:-42}"
SEEDS="${SEEDS:-$SEED}"     # space-separated; >1 seed adds _s<seed> to filenames
SET="${SET:-core}"         # core (12 categories) | vars (kick/clap/hat variations)
STEPS="${STEPS:-8}"
CFG="${CFG:-1.0}"
NEG="${NEG:-reverb, room, echo, music, loop, multiple hits, vocals, noise floor}"
ONLY="${ONLY:-}"
RUN="${RUN:-$(date +%Y%m%d-%H%M%S)_${SET}_${MODEL}}"
OUT="$HERE/renders/$RUN"

case "$MODEL" in
  sm-sfx) DECODER=same-s ;;
  medium) DECODER=same-l ;;
  *) echo "MODEL must be sm-sfx or medium" >&2; exit 1 ;;
esac

[[ -x "$SA3_DIR/sa3" ]] || { echo "sa3 not found at $SA3_DIR (set SA3_DIR)" >&2; exit 1; }
mkdir -p "$OUT"

PREFIX="TrackType: SFX,"
SUFFIX="one shot, single hit, no loop, dry, no reverb, isolated, studio quality"

# name | seconds | descriptor
PROMPTS=(
  "kick|1|punchy house kick drum, tight low thump, short click attack, fast decay"
  "808|2|deep 808 bass drum, long sub boom, smooth sine tail, slight pitch drop"
  "clap|1|crisp electronic hand clap, layered snap, short tail"
  "snare|1|tight acoustic snare drum hit, bright crack, snappy wires"
  "hat_closed|1|closed hi-hat, very short tick, bright metallic"
  "hat_open|1|open hi-hat, sizzling metallic wash, medium decay"
  "rim|1|rimshot click, woody sharp transient, very short"
  "perc_conga|1|single conga slap, warm hand drum tone"
  "perc_cowbell|1|analog cowbell hit, two-tone metallic ping"
  "shaker|1|single shaker shake, short grainy rattle"
  "riser|4|white noise riser sweep, building tension, rising filter"
  "stab|2|acid house synth chord stab, short resonant bright chord"
)

VARS=(
  "kick_big|2|huge boomy kick drum, heavy sub weight, hard beater click, long round tail"
  "kick_small|1|small tight kick drum, short dry thud, little sub, quick decay"
  "kick_house|1|classic house kick drum, punchy drum machine kick, firm click, medium decay"
  "kick_deep|1|deep house kick drum, soft rounded attack, warm low thump"
  "kick_hard|1|hard techno kick drum, overdriven distorted punch, aggressive"
  "clap_house|1|classic house hand clap, drum machine clap, layered flams, short tail"
  "clap_tight|1|single tight dry hand clap, short and snappy"
  "clap_big|1|big wide layered clap, thick body, short bright tail"
  "clap_snap|1|finger snap clap layer, crisp click, very short"
  "clap_lofi|1|lo-fi dusty sampled clap, gritty vinyl texture"
  "hat_closed_house|1|classic house closed hi-hat, crisp drum machine hat, short"
  "hat_closed_tick|1|very tight closed hi-hat tick, tiny bright click"
  "hat_closed_lofi|1|dusty sampled closed hi-hat, soft lo-fi texture"
  "hat_open_house|1|classic house open hi-hat, offbeat drum machine open hat, medium sizzle"
  "hat_open_long|2|long washy open hi-hat, bright shimmering decay"
  "hat_909_closed|1|TR-909 closed hi-hat, crunchy sampled metallic hat, bright and gritty, short"
  "hat_909_open|1|TR-909 open hi-hat, crunchy sampled metallic sizzle, bright, medium decay"
  "hat_808_closed|1|TR-808 closed hi-hat, analog square-wave metallic tick, thin chiffy tss, short"
  "hat_808_open|1|TR-808 open hi-hat, analog metallic tss, soft thin sizzle, medium decay"
)

case "$SET" in
  core) LIST=("${PROMPTS[@]}") ;;
  vars) LIST=("${VARS[@]}") ;;
  *) echo "SET must be core or vars" >&2; exit 1 ;;
esac
nseeds=$(wc -w <<<"$SEEDS")

for row in "${LIST[@]}"; do
  IFS='|' read -r name secs desc <<<"$row"
  if [[ -n "$ONLY" && ",$ONLY," != *",$name,"* ]]; then continue; fi
  prompt="$PREFIX $desc, $SUFFIX"
  for seed in $SEEDS; do
  file="$name.wav"; [[ $nseeds -gt 1 ]] && file="${name}_s${seed}.wav"
  wav="$OUT/$file"
  echo "==> $file ($secs s): $prompt"
  args=(--prompt "$prompt" --dit "$MODEL" --decoder "$DECODER"
        --seconds "$secs" --steps "$STEPS" --seed "$seed" --cfg "$CFG" --out "$wav")
  neg=""
  if [[ "$CFG" != "1.0" && "$CFG" != "1" ]]; then args+=(--negative-prompt "$NEG"); neg="$NEG"; fi
  t0=$(date +%s)
  (cd "$SA3_DIR" && ./sa3 "${args[@]}")
  t1=$(date +%s)
  python3 -c '
import json, sys
k = ["file","name","prompt","model","decoder","seconds","steps","seed","cfg","negative_prompt","render_s"]
r = dict(zip(k, sys.argv[2:]))
for f in ("seconds","cfg"): r[f] = float(r[f])
for f in ("steps","seed","render_s"): r[f] = int(r[f])
r["negative_prompt"] = r["negative_prompt"] or None
open(sys.argv[1], "a").write(json.dumps(r) + "\n")
' "$OUT/meta.jsonl" "$file" "$name" "$prompt" "$MODEL" "$DECODER" "$secs" "$STEPS" "$seed" "$CFG" "$neg" "$((t1 - t0))"
  done
done

echo
echo "Done: $OUT"
