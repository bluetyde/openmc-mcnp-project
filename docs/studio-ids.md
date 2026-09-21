# Studio object links in MCNP comments

Studio's generated `model.py` defines `studio_ids`: actual OpenMC IDs mapped to
stable Studio object IDs. The worker passes this optional dictionary to
`remediate(..., studio_ids=...)`. XML-only exports have no Studio identity data.

When identity data is provided, the runnable deck includes `c @studio-map-v1`
after its title. Individual records have this form:

```
c @studio-v1 tally 4 1/1 <payload>
```

The fields are card category (`cell`, `surface`, `material`, `tally`, or `data`),
exported card number (or data-card name), chunk number/total, and payload. The
payload is compact JSON percent-encoded as ASCII and split into 64-character
chunks, keeping every physical comment below 128 columns. Join all chunks
before decoding. The JSON contains `kind` and `id`; original primitive surfaces
can also carry a request-local `slot` for their edit descriptor.

Tally identity is recorded when each F/FMESH card is generated, including each
surface-current bin and each score. Lattice-generated cells and surfaces use
the lattice group's stable ID. Macrobody surfaces and shared surfaces have no
direct edit descriptor. Human-readable labels remain in separate comments.

Studio accepts records in full-line `c`/`C` comments or after `$`. Records
explicitly identify their card; they never depend on adjacency, labels, or
continuation-line position. Ordinary comments and data are preserved. Duplicate,
incomplete, malformed, unknown-object, and missing mappings remain read-only in
live Studio. The mapping is for navigation/edit provenance, not authentication
or a substitute for geometry and physics validation.

Regenerate the deck after model changes. Translation caching retains geometry
only; every remediation pass receives the current identity dictionary.
