# An empty pgvector read never gives its connection back

**Status: pre-existing upstream behaviour, independent of the partitioning
work, proposed as a standalone fix.**

This is the most immediately actionable of the three findings that came out of
benchmarking `document_chunk`, and it has nothing to do with partitioning. It is
one commit against `dev`, nine lines in one file.

---

## 1. What happens

`PgvectorClient.search`, `.query` and `.get` each return early when the result
set is empty, and that early return skips the `self.session.rollback()` their
populated paths take:

```python
results = result_proxy.all()

ids = [[] for _ in range(num_queries)]
...
if not results:
    return SearchResult(ids=ids, ...)      # <-- no rollback

for row in results:
    ...
self.session.rollback()  # read-only transaction
return SearchResult(...)
```

The session keeps its transaction open, and with it the connection it checked
out of the pool.

## 2. Why it stays invisible until it isn't

One thread reusing one session never notices: the next statement runs on the
same still-open transaction, and the session is eventually rolled back by
something else.

`query_collection` does not work that way. It fans out over
`asyncio.to_thread`, and `self.session` is a `scoped_session` — thread-local by
construction, so **every thread gets its own Session**. A thread that finishes
while still holding a connection never gives it back; the Session becomes
garbage and the connection is released only whenever the collector gets to it.

### The direction of causation, settled by experiment

The obvious objection is that this is backwards: an exhausted pool causes
checkout timeouts, and the timeouts are what produce the empty results. It is
worth ruling out explicitly, because the two hypotheses make opposite
predictions at low concurrency.

Three threads against a pool of fifteen, where a checkout timeout is
impossible. The only variable is whether the searched collection has rows:

| | result | checked out | free |
|---|---|---|---|
| **before the fix** | | | |
| 3 threads, **non-empty** | 10, 10, 10 | 0 | 3 |
| 3 threads, **empty** | 0, 0, 0 | **3** | **0** |
| 3 threads, non-empty again | 10, 10, 10 | 3 | 3 |
| **after the fix** | | | |
| 3 threads, non-empty | 10, 10, 10 | 0 | 3 |
| 3 threads, **empty** | 0, 0, 0 | **0** | 3 |
| 3 threads, non-empty | 10, 10, 10 | 0 | 3 |

Same concurrency, same threads, same everything — only the emptiness of the
result differs, and only the empty case retains its connections.

Three things follow. **No timeout can have occurred**: three threads, fifteen
connections, three of them free immediately beforehand. **A timeout would not
produce an empty result anyway**: it raises, and the `except` branch rolls back
and returns `None`, not a `SearchResult`; the probe shows real `SearchResult`s
with empty id lists, because the collection genuinely has no rows. And the third
row is the signature of a leak rather than of exhaustion — **three connections
checked out *and* three free at the same time**: the leaked ones stay out while
newly used ones come back normally. An exhausted pool would have none free.

The control is the last block: the only change is the added `rollback()`, and
the empty case goes from three retained connections to zero.

At scale the exhaustion does follow, and that is the second act rather than the
first. Twelve threads issuing one empty search each leave **12 connections
checked out and 0 free**; the next burst waits out the 30-second checkout
timeout and fails with `QueuePool limit of size 5 overflow 10 reached`. The
backend has stopped answering vector searches, and nothing in the application
log says why.

## 3. Why an empty result is not an exotic case

This is the part that makes it worth fixing rather than filing.

An empty result is what a filtered index scan returns when **none** of the
candidates it walked belong to the collection being searched. pgvector applies
`WHERE collection_name = …` after the ANN index walk, so on a large shared
`document_chunk` a query scoped to one knowledge base routinely comes back with
fewer rows than asked for — and sometimes with none at all.

So the searches most likely to return nothing are exactly the ones on a large,
busy install. The failure mode compounds: the bigger the table, the more empty
results, the faster the pool drains.

It is also reachable without any of that. An empty knowledge base, a file whose
chunks were deleted, a metadata filter that matches nothing — each one leaks a
connection per thread.

## 4. How it was found

Not by reading the code. A benchmark step that issues ten concurrent
per-collection searches — the shape `query_collection` produces when a user
selects ten knowledge bases — failed at 1M and 3M rows with pool timeouts, and
only in the configuration where searches come back empty. The measured latency
was 30 051 ms, which is the checkout timeout rather than anything the database
did.

The shape of that failure is itself evidence. The N=1 and N=3 blocks ran
immediately before, reported normal latencies and normal recall, and logged no
errors at all — while quietly losing one connection per dead thread. By the time
the N=10 block started there was nothing left in the pool. Nothing failed until
everything did.

## 5. The fix

Roll back before each of the three early returns. Nine lines including the
comment.

A regression test covers all three methods in both layouts: it asserts
`pool.checkedout() == 0` after a search, a query and a get against a collection
that does not exist. It fails on `dev` and passes with the fix.

## 6. What it does not fix

The recall problem that produces the empty results in the first place. That is a
separate matter, and a much larger one — see the partitioning discussion. This
change only stops an empty result from costing a connection.
