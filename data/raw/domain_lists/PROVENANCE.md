# Domain list provenance

- `moz_top500_kikobeats_20260817.json`: 500 domains, Moz Top 500 ranking, mirrored by the
  actively-maintained `Kikobeats/top-sites` GitHub repository (npm package `top-sites`).
  Repository commit used was dated 2026-08-17 (six days before data collection), so this
  is treated as the current, actively-refreshed seed list.
  Source: https://github.com/Kikobeats/top-sites

- `legacy_top10000_zer0h_2016.txt`: 10,000 domains from a 2016 Alexa-ranking snapshot
  (`zer0h/top-1000000-domains` on GitHub). This list is used only as a large *candidate
  name pool* to widen domain diversity beyond the 500 curated names above; it is NOT used
  as evidence of current popularity. Every candidate drawn from this pool is subjected to
  a live resolution check against real public resolvers at data-collection time, and only
  domains that return a valid, live answer on the collection date are retained. The
  ranking's age therefore has no bearing on the realism of the collected DNS traffic:
  what is measured is today's live response from today's authoritative infrastructure,
  for a domain name that happened to be sourced from an older popularity snapshot.
  Source: https://github.com/zer0h/top-1000000-domains
