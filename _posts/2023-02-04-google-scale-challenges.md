---
title: Tech talk notes on "Building Software Systems At Google and Lessons Learned"
date: 2023-02-04 00:00:00 Z
layout: post
---

Notes from [Building Software Systems At Google and Lessons Learned](https://www.youtube.com/watch?v=modXC5IWTJI)

* Google search service - indexing service and document service
* Indexing service handles queries by looking up some inverted queries and outputting some form of (document id, word offset). This tuple then used by document service
* Indexing is the bottleneck and requires pre-prociessing and carefully handling the request
* Spread out reads to multiple replication machines , load balancing
* Also increase index size so sharding to cover more data
* Query size increase - google moved to an in-memory indexing scheme instead to avoid disk seeks
* In-memory index lightweight encoding schemes less cpu intensive - compression still required to have it fit in memory. 
* Encoding scheme like variable length encoding. 4 32 bit words indexed by a 1 byte prefix byte which indicates how many bytes required per 32 bits. 
* Wrote map reduce to simplify the indexing tasks
* Think about back of envelopes carefully.
* Design pattern mentioned - 
single master, 1000s of workers : easier to reason, deisgned such that minimal overhead with master, spof, used in GFS, bigtable, mapreduce clusters
* Tree based query distribution - avoids root fan in on single root machine RPC calls. Parent sends requests to leaves, limits the fan out helps with network congestion. Structure allows heirarchical data properties to reduce network utilization. For instance instead of each machine shipping the 10 best queires and have one machine combine that, each level filters out 10 queries less traffic
* Backup machines to minimize latency - mapreduce
* Smaller units per machine. Faster recovery and better load distribution
* Elastic system for instance web search overloaded system reduce the index space searched QOS suffers but still reasonable, spelling service dropped when high latency
