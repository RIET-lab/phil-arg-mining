

Prioritized:
1. Determine the avg token length of workshop annotations (for snowball.phase_1.hparams.end2end.decoding.max_new_tokens)
2. Determine chunk size based on average ADU token length
3. Determine chunk overlap boundary/hparam based on average ADU token length
4. Determine top-k RAG results based on how many ADU candidates we expect
5. Update metadata logic where if it's a directory, takes the most recent file, but if its a file, just takes the file.
6. Implement processes described in __readme__s.

Unprioritized:
- Determine if we can safely remove grobid references
- Refactor mere_workshop_listener (cleanup scripts)
- Refactor mere_workshop (remove unnecessary code)
- Create a join text/metadata philpapers.csv master dataset with original and revised versions. (Revised uses the langdetect package to filter out non-english samples).
- N. clean entry point scripts in `scripts` directory
