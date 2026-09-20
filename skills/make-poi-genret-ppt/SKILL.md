---
name: make-poi-genret-ppt
description: Create or edit POI generative-retrieval project PowerPoint files (.pptx) in the established Hu Dan 0727/0810 phase-report style. Use for new progress decks, extending an existing report, converting experiment records into slides, or restyling draft slides. Use tables for experiment results and flows or architectures for methods, with uniformly black major titles. Enforce template reuse, restrained colors, no decorative boxes or unsupported square glyphs, evidence-backed metrics from docs/outputs, and before/after PPTX validation.
---

# POI GenRet 汇报 PPT

## Establish the source of truth

1. Read the repository `AGENTS.md`, `README.md`, `方案.md`, `docs/PROJECT_STATUS.md`, and the relevant `docs/experiments/*.md` files. For experiment content, also follow `skills/poi-genret-workflow/SKILL.md`.
2. Read [references/style-guide.md](references/style-guide.md) completely.
3. Inspect both canonical decks on every PPT task:
   - `/ofs/map_search/hudan/poi_genret/PPT/胡丹-0727.pptx`
   - `/ofs/map_search/hudan/poi_genret/PPT/胡丹-0810.pptx`
4. Use `0810` for visual density and restrained formatting, and `0727` for the original template language and reusable layouts. The current rules below override historical examples: major titles are black, results use tables, and methods use flows/architectures. If the user supplies a newer deck or PDF, treat that artifact as the content authority.
5. Edit the matching PPTX. Use a PDF only for visual/page comparison; do not rebuild from PDF when an editable PPTX exists.

Before editing, state the current step, what will not change, and the acceptance criteria. Do not modify experiments or project documents merely to make a slide.

## Build evidence-backed content

- Use formal experiment documents as the primary narrative source and `outputs/` metrics, manifests, and logs for verification.
- Never infer missing metrics or describe a stopped/failed run as successful. Distinguish Trainer success from an outer platform-wrapper failure.
- Keep comparison cells protocol-compatible. State important limitations such as fixed 10k Validation, method-specific constrained decoding, or missing Test results.
- Lead each slide with one conclusion. Choose the visual by content: tables for results, dataset statistics and exact comparisons; flows for method steps, architectures for module interactions, and timelines for training stages. Use concise prose for interpretation and simple conclusions. Do not turn a method into a table merely because its description has multiple parts.
- Speaker notes are not part of the default deliverable. Do not create, expand, or rewrite notes unless the user explicitly asks for notes in that task; simplify or split overloaded visible content instead.

## Preserve the validated visual language

1. Start from the closest existing slide in `0810` or `0727`; duplicate and replace content instead of inventing a new layout.
2. Preserve the 16:9 slide size, master, theme, title position, page-number treatment, margins, fonts, and restrained palette.
3. Set every major title, including cover and section titles, explicitly to black (`#000000`); do not inherit a historical navy title. Keep method content flow-first: show inputs, transformations, outputs, and meaningful branches or feedback. Use native editable PowerPoint objects and connectors.
4. Do not surround ordinary text sections with rectangles or make decorative card grids. Minimal white/light-gray process nodes with thin gray lines are allowed when they represent actual modules or states; arrows must express real data flow, dependencies or sequence. Do not add black-outline boxes, decorative frames, colored badges, gradients, shadows, stickers, emoji, or ornamental icons.
5. Do not use uncommon Unicode bullets or symbols that can render as small black outlined squares. Use plain Chinese text, standard punctuation, or the template's native bullet formatting.
6. Reuse the table style from `0810`: quiet gray header, white/light alternating rows, and at most one pale-blue highlight for the selected result.
7. Keep the deck visually calm. Do not introduce a new accent color merely to differentiate methods.

For exact palette, typography, table, layout, and notes rules, follow [references/style-guide.md](references/style-guide.md).

## Apply task-specific editing rules

### Create or extend slides

- Reuse the nearest slide structure and remove unused objects instead of hiding them outside the canvas.
- Keep one primary table or one primary visual per results slide when possible.
- Method pages should make direction and dependencies visible, not repackage table rows as disconnected boxes. A simple explanation or summary can remain plain text; not every page needs a diagram.
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

3. Inspect every slide visually when rendering is available. Check clipping, overlap, inconsistent alignment, tiny text, excess colors, decorative outlined rectangles, and tofu/square glyphs. Verify all major titles are black and method flows have readable direction, functional connectors and no table-like card grid. If only a geometry preview is available, disclose that it is not an Office rendering.
4. For notes-only work, require canonical slide body hashes and shape counts to match the baseline.
5. Report the output path, page count, substantive additions, validation result, current limitation, and commit/push status.

Do not commit or push unless explicitly requested.
