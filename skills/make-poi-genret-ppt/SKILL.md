---
name: make-poi-genret-ppt
description: Create or edit POI generative-retrieval project PowerPoint files (.pptx) in the established Hu Dan 0727/0810 phase-report style. Use for new progress decks, extending an existing report, converting experiment records into slides, or restyling draft slides. Enforce template reuse, restrained colors, table-first experiment reporting, no decorative boxes or unsupported square glyphs, evidence-backed metrics from docs/outputs, and before/after PPTX validation.
---

# POI GenRet 汇报 PPT

## Establish the source of truth

1. Read the repository `AGENTS.md`, `README.md`, `方案.md`, `docs/PROJECT_STATUS.md`, and the relevant `docs/experiments/*.md` files. For experiment content, also follow `skills/poi-genret-workflow/SKILL.md`.
2. Read [references/style-guide.md](references/style-guide.md) completely.
3. Inspect both canonical decks on every PPT task:
   - `/ofs/map_search/hudan/poi_genret/PPT/胡丹-0727.pptx`
   - `/ofs/map_search/hudan/poi_genret/PPT/胡丹-0810.pptx`
4. Treat `0810` as the final visual-density and formatting authority. Use `0727` for the original template language and reusable layouts. If the user supplies a newer deck or PDF, treat that artifact as the content authority while retaining the established style.
5. Edit the matching PPTX. Use a PDF only for visual/page comparison; do not rebuild from PDF when an editable PPTX exists.

Before editing, state the current step, what will not change, and the acceptance criteria. Do not modify experiments or project documents merely to make a slide.

## Build evidence-backed content

- Use formal experiment documents as the primary narrative source and `outputs/` metrics, manifests, and logs for verification.
- Never infer missing metrics or describe a stopped/failed run as successful. Distinguish Trainer success from an outer platform-wrapper failure.
- Keep comparison cells protocol-compatible. State important limitations such as fixed 10k Validation, method-specific constrained decoding, or missing Test results.
- Lead each slide with one conclusion. Use tables for repeated metrics and exact mappings; use prose only for interpretation.
- Speaker notes are not part of the default deliverable. Do not create, expand, or rewrite notes unless the user explicitly asks for notes in that task; simplify or split overloaded visible content instead.

## Preserve the validated visual language

1. Start from the closest existing slide in `0810` or `0727`; duplicate and replace content instead of inventing a new layout.
2. Preserve the 16:9 slide size, master, theme, title position, page-number treatment, margins, fonts, and restrained palette.
3. Prefer text plus tables. Use a diagram only when relationships cannot be understood from a short paragraph or table.
4. Do not surround sections with rectangles. Do not add card grids, black-outline boxes, decorative frames, colored badges, gradients, shadows, stickers, emoji, or ornamental icons.
5. Do not use uncommon Unicode bullets or symbols that can render as small black outlined squares. Use plain Chinese text, standard punctuation, or the template's native bullet formatting.
6. Reuse the table style from `0810`: quiet gray header, white/light alternating rows, and at most one pale-blue highlight for the selected result.
7. Keep the deck visually calm. Do not introduce a new accent color merely to differentiate methods.

For exact palette, typography, table, layout, and notes rules, follow [references/style-guide.md](references/style-guide.md).

## Apply task-specific editing rules

### Create or extend slides

- Reuse the nearest slide structure and remove unused objects instead of hiding them outside the canvas.
- Keep one primary table or one primary visual per results slide when possible.
- Split dense content across slides before reducing body text below the established template size.
- Preserve user-edited wording unless the user explicitly requests copyediting.

### Add notes only when explicitly requested

- Do not enter this mode from an ordinary PPT creation, extension, or revision request.
- Change only speaker-note parts and required OOXML relationships.
- Do not alter slide text, tables, pictures, shapes, geometry, colors, masters, or ordering.
- Capture canonical hashes and shape counts before editing; compare them after saving.
- Use the repository audit script with `--baseline <before.pptx> --require-notes <after.pptx>` as the final gate.

### Revise an existing deck

- Treat the user's latest file as authoritative even when an older deck is cleaner.
- Make the smallest requested change. Do not opportunistically redesign other pages.
- If the user points out a visual defect, inspect the actual OOXML object or character causing it before changing unrelated formatting.

## Validate before delivery

1. Reopen the PPTX and verify slide count, slide order, and ZIP integrity. Verify notes only when the user explicitly requested them.
2. Run:

   ```bash
   /ofs/map_search/hudan/envs/poi-gr/bin/python \
     skills/make-poi-genret-ppt/scripts/audit_pptx.py \
     --reference /ofs/map_search/hudan/poi_genret/PPT/胡丹-0810.pptx \
     <output.pptx>
   ```

3. Inspect every slide visually when rendering is available. Check clipping, overlap, inconsistent alignment, tiny text, excess colors, outlined rectangles, and tofu/square glyphs.
4. For notes-only work, require canonical slide body hashes and shape counts to match the baseline.
5. Report the output path, page count, substantive additions, validation result, current limitation, and commit/push status.

Do not commit or push unless explicitly requested.
