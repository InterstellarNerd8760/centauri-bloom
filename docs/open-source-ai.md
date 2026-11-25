# Open Source AI release policy

Centauri Bloom adopts the [Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition) (OSAID), published by the Open Source Initiative, as its target for official model releases. The project is currently a research scaffold; adopting this policy is not a claim of compliance or OSI certification.

## Licensing

- Original repository code and documentation: Apache-2.0, unless explicitly identified otherwise.
- Official model weights: prefer Apache-2.0 or other compatible open terms after confirming the right to release them; record the terms in each model card.
- Original datasets: prefer CC0 or CC BY 4.0 where appropriate and authorized. Dataset license selection is separate from code licensing.
- Third-party material: retain applicable license and attribution notices; confirm compatibility and redistribution rights before inclusion. Do not assume that public availability means permission.
- Names and logos: no additional trademark permissions are granted by the code license.

Contributors must have the rights to license their submissions. Independently developed browser tools, weights and datasets are not automatically relicensed by this repository.

## Release checklist

Every official model release must be reviewed against OSAID 1.0. The release must enable use, examination, modification and redistribution for any purpose. No additional field-of-use restrictions may be imposed on an OSAID-conforming release.

- [ ] Publish the complete source for the architecture, training, inference, data processing/filtering, tokenizer, evaluation and any other components used to produce and run the model, under OSI-approved licenses.
- [ ] Publish sufficient training-data information under OSI-approved terms to permit a skilled practitioner to reconstruct a substantially equivalent system: provenance, composition, acquisition and selection, labeling, processing, filtering and deduplication, including any non-redistributable data. List public and third-party data sources and how to obtain them.
- [ ] Prefer fully redistributable datasets; publish the datasets or immutable retrieval instructions and record their licenses and privacy review. Do not publish material without permission.
- [ ] Publish modifiable weights, architecture configuration and tokenizer assets under OSI-approved terms. Retain and publish relevant training checkpoints and final optimizer state where needed for modification.
- [ ] Publish training commands, settings, dependency versions, dataset versions, seeds, evaluation methods and measured results, along with known limitations.
- [ ] Link the corresponding Git commit, dataset card, model card, artifact hashes and license/attribution notices from the release.
- [ ] Record a maintainer review of the applicable [OSI checklist](https://opensource.org/ai/open-source-ai-definition) and any remaining gaps. Label incomplete artifacts as experimental, not as conforming official releases.

GitHub is the canonical home for code and release decisions. Hugging Face is the planned distribution platform for reviewed model and dataset artifacts. A downloadable GGUF or a permissive license alone does not establish OSAID compliance.
