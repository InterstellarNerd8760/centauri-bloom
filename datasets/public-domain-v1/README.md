---
language:
  - en
license: other
license_name: Public domain in the United States
license_link: https://www.gutenberg.org/policy/license.html
task_categories:
  - text-generation
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-00000.jsonl
      - split: validation
        path: data/validation-00000.jsonl
      - split: test
        path: data/test-00000.jsonl
---

# Centauri Bloom public-domain English corpus

Version 1.0.0 contains eight English-language literary works by eight authors, first published from 1820 to 1908. It was selected for small next-byte language-model experiments from random initialization. The source download is about 1.9 MB. It offers varied sentence lengths, dialogue, and punctuation while keeping the whole corpus within reach of CPU contributors. Each work forms one unlabeled document.

## Source editions

[sources.json](sources.json) pins one GITenberg mirror file per work by Git commit and raw SHA-256. Each file is a Project Gutenberg English ebook. The edition metadata at the same Git commit identifies each selected work as public domain in the USA.

| Dataset source | Work | Author | First published | Pinned edition metadata |
| --- | --- | --- | ---: | --- |
| `pg-11` | Alice's Adventures in Wonderland | Lewis Carroll | 1865 | [Ebook 11](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland/blob/7f18748e114b994d5d9454d5793581bc0b9cf9d7/metadata.yaml) |
| `pg-41` | The Legend of Sleepy Hollow | Washington Irving | 1820 | [Ebook 41](https://github.com/GITenberg/The-Legend-of-Sleepy-Hollow_41/blob/5f864ee8e6fabb0167d60cc625e55da67830854a/metadata.yaml) |
| `pg-43` | The Strange Case of Dr. Jekyll and Mr. Hyde | Robert Louis Stevenson | 1886 | [Ebook 43](https://github.com/GITenberg/The-Strange-Case-of-Dr.-Jekyll-and-Mr.-Hyde_43/blob/534f1e1d3193ffaf351570cf808e0ee6db70efe9/metadata.yaml) |
| `pg-46` | A Christmas Carol | Charles Dickens | 1843 | [Ebook 46](https://github.com/GITenberg/A-Christmas-Carol_46/blob/b173bd3787848ecbded8d036ed9f5326c738da19/metadata.yaml) |
| `pg-289` | The Wind in the Willows | Kenneth Grahame | 1908 | [Ebook 289](https://github.com/GITenberg/The-Wind-in-the-Willows_289/blob/9032846214f647204585df2572cd19dd7fbc3c73/metadata.yaml) |
| `pg-550` | Silas Marner | George Eliot | 1861 | [Ebook 550](https://github.com/GITenberg/Silas-Marner_550/blob/a4c56c3b0a355f69ceba3fb73adf25579ad3ddd9/metadata.yaml) |
| `pg-1837` | The Prince and the Pauper | Mark Twain | 1881 | [Ebook 1837](https://github.com/GITenberg/The-Prince-and-the-Pauper_1837/blob/e425ef5d138845955c87b4b07b683c7d019d1d0c/metadata.yaml) |
| `pg-1952` | The Yellow Wallpaper | Charlotte Perkins Gilman | 1892 | [Ebook 1952](https://github.com/GITenberg/The-Yellow-Wallpaper_1952/blob/ce7c5bc1a5cf0fe23bcb265545a900365dbae29c/metadata.yaml) |

## Licensing and privacy review

The included literary text is public domain in the United States. Public-domain status varies by jurisdiction. The upstream inputs retain the complete ebook notices and terms. The generated text shards omit the ebook wrapper, trademark references, and modern transcriber credits. This follows the distinction between the public-domain work and the branded ebook in sections 1.C and 1.D of the source license. The source lock and this card record provenance separately from the training text. Repository code and documentation are Apache-2.0. [Project Gutenberg license and redistribution terms](https://www.gutenberg.org/policy/license.html)

The privacy review checked the source headers and extraction boundaries and scanned the retained text for conventional email addresses, web links, North American phone number patterns, and digitizer credit keywords. These checks returned zero matches. Selection was limited to historic published fiction, with no contemporary submissions, account records, or web crawl. Author names, character names, historical references, and place names may occur in the text. The checks cover common contact formats and do not certify that every possible personal reference was identified.

## Processing and limitations

The builder verifies the raw hash before decoding the source as UTF-8, removes any UTF-8 BOM, converts CRLF and CR line endings to LF, and trims surrounding ASCII whitespace. The exact body boundary lines are recorded in the source lock. They remove the ebook header, license and footer, modern credit blocks, and the digital edition label for Alice. The rest of the selected span keeps its spelling, punctuation, wrapping, and any in-span headings, prefaces or notes.

Exact duplicate documents are removed globally before deterministic assignment of whole works to train, validation, and test. Retaining whole works keeps adjacent passages from crossing splits. Shared phrases and near-duplicate passages can remain. A corpus this small can measure whether an experiment runs end to end. Held-out losses are sensitive to the selected authors and book lengths. The collection emphasizes British and U.S. historical fiction and may contain dated stereotypes, offensive language, and unsettling themes.

The version 1.0.0 build with split seed 0 yielded 1,679,306 byte tokens.

| Split | Documents | Byte tokens |
| --- | ---: | ---: |
| Train | 6 | 1,188,671 |
| Validation | 1 | 332,220 |
| Test | 1 | 158,415 |

The SHA-256 of the resulting `manifest.json` is `d92f0164dba7481f21202faf6b04299a936796674f78c7b1a0538da857724518`.

## Distribution layout

The builder writes one UTF-8 JSON Lines shard per split under `data/`, with `id` and `text` columns. The repository's dataset guide gives the build and verification commands. [Dataset guide](../../docs/text-dataset.md)

A future Hugging Face release can place this card and `sources.json` at the root alongside `data/`, `manifest.json`, and `manifest.sha256` after release review. Update the repository documentation links to point to the release commit when copying the card. Check the source terms and applicable jurisdictions before redistributing any artifacts. [Centauri Bloom release policy](../../docs/open-source-ai.md)
