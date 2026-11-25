# Centauri Bloom — Design system

Status: proposed v0.1 · Community review encouraged

## Direction

A small, open AI research laboratory: editorial, botanical, technical and understated.
All Bloom websites and training tools should feel related without requiring identical
implementations. The existing [project page](https://westonrosch.com/projects/centauri-bloom/)
is the visual baseline. Preserve its dark surfaces, serif headlines, restrained borders
and geometric flower mark. Green is the primary brand color; pink is a selective accent.

## Color tokens

```css
:root {
  --bloom-bg: #05080f;
  --bloom-panel: #0a1220;
  --bloom-surface: #0f1a2b;
  --bloom-green: #8fd1ae;
  --bloom-pink: #f2a7cf;
  --bloom-text: #c9d3de;
  --bloom-heading: #f2f5f8;
  --bloom-muted: #a1afbf;
  --bloom-border: rgba(160, 185, 215, .14);
  --bloom-border-strong: rgba(160, 185, 215, .26);
}
```

The dark neutrals and green are inherited from the current project site. Pink and the
explicit muted token are proposed additions. Use green for primary actions, links and
the mark. Pink may emphasize selected highlights, secondary charts or experimental
features; do not use it as the only signal for a state. Use dark text on filled
green or pink buttons. Keep status colors semantically distinct.

## Typography

- Headings and wordmark: Instrument Serif, Georgia, serif.
- Interface and body: IBM Plex Sans, system-ui, sans-serif.
- Technical metadata, metrics and code: IBM Plex Mono, monospace.
- Body: 16px, line height 1.6–1.65; headings scale responsively.
- Small uppercase monospace labels with restrained letter spacing.

## Identity

Use the existing six-petal geometric outline flower with a central circle and optional
dotted orbital ring. The mark should work as a monochrome SVG and at favicon size
(with the dotted ring omitted if necessary). Use the text “Centauri Bloom” on
public-facing pages to distinguish the project from unrelated models.
Use “Bloom Text” and “Bloom Image” as descriptive experiment labels, pending
community agreement. Avoid independently redesigning the logo for each tool.

## Layout and components

- Page container: up to 78rem; reading content around 60rem or narrower.
- Spacious editorial sections with simple hierarchy and 1px borders.
- Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64px.
- Panels and buttons: restrained 6px corner radius; avoid heavy shadows and gradients.
- Primary buttons: green fill with dark text. Secondary: outline or text link.
- Technical dashboards: model name, checkpoint, dataset version, metrics and controls
  should be easy to find. Show units, progress, errors and an explicit stop control.
- Mobile: single-column layout as needed; avoid horizontal overflow; usable controls
  and readable plots. Respect reduced motion, keyboard navigation and visible focus.
- Follow WCAG AA contrast guidelines, test actual component combinations, and never
  rely on color alone to communicate status.

## Shared navigation

Use only the links a particular tool needs: project overview, training, GitHub,
Hugging Face (once established) and Discord. Keep the canonical destinations in one
place. Link back to the main project page from independent tools. Do not duplicate
the entire project overview on every site.

## Implementation

Copy the CSS tokens or map them to the framework's own theme. Keep local code and
hosting independent while treating this file as the common visual reference.
Propose changes to this document through a GitHub issue or pull request. A shared
domain is optional and does not block development.

## Content principles

Distinguish prototypes from reviewed releases. Report actual metrics and limitations.
Use model and dataset cards, credit contributors, and link to reproducible code.
Do not imply affiliation with BigScience BLOOM. Avoid collecting unnecessary personal
information or credentials in training tools.
