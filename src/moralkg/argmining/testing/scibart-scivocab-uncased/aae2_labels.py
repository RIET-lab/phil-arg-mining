# aae2_labels.py

from pie_datasets import load_dataset # type: ignore
from pie_datasets.builders.brat import BratDocumentWithMergedSpans # type: ignore
from pytorch_ie.documents import TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions # type: ignore

ds = load_dataset("pie/aae2")
assert isinstance(ds["train"][0], BratDocumentWithMergedSpans)
ds = ds.to_document_type(TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions)
assert isinstance(ds["train"][0], TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions)

span_labels = set()
relation_labels = set()
for doc in ds["train"]:
    for span in doc.spans:      
        span_labels.add(span.label)
    for rel in doc.binary_relations:
        relation_labels.add(rel.label)

print("Span labels in AAE2:", sorted(span_labels))
print("Relation labels in AAE2:", sorted(relation_labels))