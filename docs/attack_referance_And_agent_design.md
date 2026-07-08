# Attack Reference & Autonomous Remediation Agent Design
### Real-Time Incident Tracking & Autonomous Threat Remediation — CICIDS2017 Attack Catalogue

*Companion document to the Architecture Document and Project Charter. Use this to design DQN reward shaping, the Gymnasium environment's action outcomes, and the evaluation report's per-attack discussion.*

---

## How to read each entry

For every attack type you have in `attack_class_counts.json`, this document gives:

1. **What it is** — mechanism, at the network-flow level (ties to what your autoencoder will see as reconstruction error)
2. **Detection fingerprint** — the flow-feature signature that makes it anomalous
3. **Prevention** — controls to reduce exposure *before* an attack happens
4. **Incident response (human playbook)** — what a SOC analyst would do manually
5. **AI agent behavior** — what your DQN agent should do autonomously, mapped to your 5-action space (Block IP / Revoke Credentials / Isolate Server / Kill Process / Monitor), plus reward-shaping notes

---

## 1. DoS Hulk (297,642 rows — 49.6%)

**What it is:** HTTP Denial-of-Service tool that floods a web server with a high volume of GET/POST requests using randomized, obfuscated HTTP headers to bypass simple caching and rule-based filters. Goal is to exhaust server thread pools / connection limits.

**Detection fingerprint:** Enormous forward packet count and forward byte count in a very short flow duration; low backward traffic (server struggling to respond); many concurrent flows from few source IPs to one destination port (usually 80/443).

**Prevention:**
- Rate limiting / connection throttling per source IP at the load balancer or reverse proxy (nginx `limit_req`, HAProxy `stick-table`)
- Web Application Firewall (WAF) with anomaly-based request-rate rules
- CDN/edge absorption (Cloudflare, AWS Shield) to soak volumetric floods before they reach origin
- Auto-scaling web tier so load is distributed, buying detection time
- SYN cookies and connection queue tuning at the OS/network layer

**Incident response (manual):**
1. Confirm via server logs / APM that request rate from suspect IP(s) is abnormal (not a real traffic spike)
2. Apply temporary rate limit or null-route the source IP at the edge
3. If distributed sources, escalate to upstream ISP/CDN scrubbing
4. Restart or scale out affected web service if thread pool exhaustion already occurred
5. Post-incident: tune WAF rule to catch the specific header obfuscation pattern used

**AI agent behavior:**
- **Primary action:** `Block IP` (source is usually identifiable even in the "low-and-slow but high-volume" Hulk pattern)
- **Secondary action:** `Isolate Server` only if the web server is already unresponsive (protect it from cascading failure) — otherwise blocking IP is sufficient and less disruptive
- **Avoid:** `Kill Process` — killing the web server process itself worsens availability; only appropriate if the process is compromised, not merely overloaded
- **Reward shaping:** heavily reward `Block IP` when reconstruction error correlates with high fwd-packet-count + short duration; penalize `Isolate Server` here since it causes unnecessary service disruption (-5) unless the flood already crashed the service

---

## 2. Port Scan (161,315 rows — 26.9%)

**What it is:** Reconnaissance technique where a single source systematically probes many destination ports (or hosts) to discover open services, versions, and potential vulnerabilities before a real attack. Usually the *first* stage of a multi-stage intrusion, not damaging by itself.

**Detection fingerprint:** One source IP contacting an unusually large number of distinct destination ports/IPs in a short window; abnormal SYN/RST flag ratios (many SYN with no completed handshake, or SYN-RST pairs); very small packet sizes and short flow durations.

**Prevention:**
- Network segmentation so scans can't traverse from one subnet to sensitive assets
- Close/firewall unused ports by default (deny-by-default egress and ingress rules)
- Deploy honeypots/deception ports that trigger high-confidence alerts on first touch
- IDS/IPS scan-detection signatures (Snort/Suricata `portscan` preprocessor)

**Incident response (manual):**
1. Identify scanning source and scope of ports/hosts probed
2. Block source IP at perimeter firewall
3. Check which of the scanned services are actually exposed and patch/harden them
4. Review logs for any follow-up exploitation attempts from the same or related IPs (scans are often followed within minutes/hours by a targeted attack)

**AI agent behavior:**
- **Primary action:** `Block IP` — scanning traffic has no legitimate service dependency, so blocking is low-risk and high-value
- **Secondary action:** `Monitor` with elevated logging if the scan is very low-rate (could be a stealth scan blending with legitimate traffic — blocking too eagerly on ambiguous signals risks false positives)
- **Reward shaping:** `Block IP` → +10 (it's a clean, unambiguous confirmation case for your agent to learn on, given the huge sample count); a `false alarm` penalty (-3) should be tuned carefully here since some vulnerability scanners are legitimate (internal security team scans) — consider a source-reputation feature in the state vector

---

## 3. DDoS LOIT (95,729 rows — 15.9%)

**What it is:** Distributed Denial-of-Service using LOIC/HOIC-style tools ("Low Orbit Ion Cannon") — many distributed sources flood a target simultaneously with TCP/UDP/HTTP requests. Unlike DoS Hulk (single source), this is coordinated across a botnet.

**Detection fingerprint:** Many source IPs, each individually looking like moderate traffic, but destination sees an aggregate flood; consistent flow duration/packet-size fingerprint across sources (same tool, same config); shared destination port.

**Prevention:**
- Upstream DDoS scrubbing service (cloud-based, absorbs volumetric attacks before your network)
- Anycast network design to disperse load geographically
- BGP blackholing/RTBH for extreme volumetric floods
- Egress filtering to prevent your own infrastructure being conscripted into a botnet

**Incident response (manual):**
1. Confirm distributed nature (many sources, one destination) via NetFlow/sFlow
2. Engage upstream ISP or DDoS mitigation provider — single-box IP blocking doesn't scale to thousands of sources
3. Apply rate limiting per-flow at the edge as a stopgap
4. Communicate to stakeholders (this is often visible to customers — status page update)

**AI agent behavior:**
- **Primary action:** `Isolate Server` (or the affected service instance) if the DQN detects that per-IP blocking won't keep pace with the number of distinct attacking sources — a single autonomous agent sitting at one enforcement point usually cannot block a true DDoS one IP at a time fast enough
- **Secondary action:** `Block IP` for the top-N highest-volume sources as a stopgap while isolation/failover completes
- **Escalation:** this is the clearest case where the agent should also emit a **human escalation alert** (via WebSocket/incident record) rather than rely purely on autonomous action — DDoS mitigation often requires upstream network changes outside the agent's control
- **Reward shaping:** give partial credit for `Block IP` (reduces load, +some reward) even if it doesn't fully neutralize the attack, since perfect neutralization isn't achievable from this vantage point alone

---

## 4. FTP-Patator (9,531 rows — 1.6%)

**What it is:** Brute-force credential-guessing tool targeting the FTP service (port 21) — automated, rapid-fire username/password attempts.

**Detection fingerprint:** Many short-lived flows to port 21 from one source, high connection rate, repeated failed-auth patterns (if payload/log correlation available); low bytes-per-flow (just auth handshake, no data transfer).

**Prevention:**
- Disable FTP in favor of SFTP/FTPS wherever possible
- Account lockout after N failed attempts
- Fail2ban-style automatic IP banning tied to auth logs
- MFA on any service that must remain password-based
- IP allowlisting for administrative services

**Incident response (manual):**
1. Check auth logs for the FTP service to confirm brute-force pattern and whether any login succeeded
2. If a login succeeded: treat as a credential-compromise incident — force password reset, review file access logs
3. Block source IP
4. If success occurred, **revoke the compromised credential** immediately, not just block the IP (attacker may pivot to a different source)

**AI agent behavior:**
- **Primary action:** `Block IP` for the vast majority of cases (attempt volume alone, no successful auth)
- **Escalation action:** `Revoke Credentials` should trigger automatically if the state includes a "successful authentication after a burst of failures" signal — this is the one case where IP blocking alone is insufficient, since the attacker already has valid credentials
- **Reward shaping:** `Revoke Credentials` should carry a moderate action cost (some legitimate user disruption if it's a false positive) but a very high reward (+10) when a credential-stuffing success is correctly caught, since missing it (a false negative here) is one of the worst outcomes for the whole system

---

## 5. DoS GoldenEye (8,363 rows — 1.4%)

**What it is:** HTTP flood tool similar to Hulk but uses persistent HTTP keep-alive connections with a smaller number of long-lived sockets designed to exhaust the server's concurrent-connection limit rather than raw bandwidth.

**Detection fingerprint:** Long-duration flows (keep-alive) with cyclical small packet bursts, high connection count per source, moderate (not extreme) packet counts compared to Hulk.

**Prevention:**
- Limit max keep-alive connections per client IP at the web server/reverse proxy level
- Set aggressive keep-alive timeouts
- WAF rules tuned for slow, persistent-connection floods (different signature than volumetric floods)

**Incident response (manual):**
1. Check web server connection table for abnormal per-IP concurrent connection counts
2. Reduce keep-alive timeout / max connections config as an immediate mitigation
3. Block offending IP(s)
4. Restart web server workers if connection table is already saturated

**AI agent behavior:**
- **Primary action:** `Block IP`
- **Secondary consideration:** because this is a *connection-exhaustion* style attack (like Slowloris family below), the agent's confidence threshold should account for legitimate long-lived connections (websockets, streaming) — false positives are more likely here than with Hulk, so slightly favor `Monitor` when ambiguous, escalate to `Block IP` if error stays elevated over consecutive windows (temporal persistence, not single-flow judgment)

---

## 6. DoS Slowhttptest (6,856 rows — 1.1%)

**What it is:** "Slow HTTP" attack sending partial HTTP requests/headers at a deliberately slow rate, keeping connections open indefinitely to exhaust the server's connection pool with minimal bandwidth (very hard to detect via volume alone).

**Detection fingerprint:** Extremely long flow duration, very low packet/byte counts, unusually small inter-arrival-time variance (deliberately paced), sustained open connections.

**Prevention:**
- Set minimum data rate / request-completion timeouts on the web server (Apache `mod_reqtimeout`, nginx `client_body_timeout`)
- Limit total concurrent connections per IP
- Use reverse proxies (nginx) in front of vulnerable servers (e.g., Apache) since they buffer full requests before forwarding

**Incident response (manual):**
1. Identify connections stuck in "receiving headers" state for abnormally long periods
2. Apply/verify request timeout settings
3. Block source IP(s) holding open the slow connections
4. Review whether the origin server needs a reverse-proxy layer added

**AI agent behavior:**
- **Primary action:** `Block IP` once duration exceeds a learned threshold with near-zero throughput — this is a strong, distinctive fingerprint (duration >> typical, bytes ≈ 0) that your autoencoder should reconstruct poorly
- **Reward shaping:** since this attack is low-and-slow, the agent should be rewarded for detecting it *before* connection pool exhaustion occurs (early action on a persistent-but-low-volume anomaly), not just after — consider a time-decay bonus in the reward function for faster correct action

---

## 7. DoS Slowloris (5,122 rows — 0.85%)

**What it is:** Same family as Slowhttptest — sends partial HTTP headers very slowly, one connection at a time, to hold open many connections without ever completing a request, exhausting the server's max-connections limit.

**Detection fingerprint:** Nearly identical to Slowhttptest — long duration, minimal bytes, many concurrent flows from one source, slow/periodic keep-alive packets.

**Prevention:** Same as Slowhttptest — timeout enforcement, reverse proxy buffering, per-IP connection caps, and specifically for Slowloris: modules like `mod_antiloris` / `mod_reqtimeout` on Apache.

**Incident response (manual):** Identical playbook to Slowhttptest — timeout tuning, blocking, and adding a proxy layer if not already present.

**AI agent behavior:**
- **Primary action:** `Block IP`
- Because Slowloris and Slowhttptest are nearly indistinguishable at the flow-feature level, your DQN doesn't need to distinguish the *tool* — it should generalize a single "slow connection exhaustion" policy that applies to both. This is a good opportunity to **merge these two into one semantic action category** in your reward design rather than treating them as fully separate states.

---

## 8. SSH-Patator (5,949 rows — 0.99%)

**What it is:** Brute-force credential guessing against SSH (port 22) — same mechanism as FTP-Patator, different target service. Often a precursor to full server compromise since SSH grants shell access.

**Detection fingerprint:** High connection rate to port 22 from one source, short flows (auth handshake only), repeated attempts in rapid succession.

**Prevention:**
- Disable password auth, require key-based SSH authentication only
- Move SSH off the default port (security-through-obscurity, minor but reduces automated scan noise)
- `fail2ban` / `sshguard` for automatic temporary banning
- Bastion host / VPN-gated access instead of direct internet-facing SSH
- MFA for SSH where supported

**Incident response (manual):**
1. Check `/var/log/auth.log` (or equivalent) for successful logins following the brute-force burst
2. If any succeeded: treat as a confirmed compromise — isolate the host, rotate all credentials/keys, forensically image if needed
3. Block source IP
4. Audit `authorized_keys` and user accounts for unauthorized additions

**AI agent behavior:**
- **Primary action:** `Block IP` for the attempt volume
- **Escalation action:** if state indicates a successful login after brute-force pattern, this is one of the highest-severity cases in your whole action space — SSH compromise implies shell access. The agent should trigger **both** `Revoke Credentials` *and* `Isolate Server` (a compound/escalated response), since a compromised SSH session can pivot laterally
- **Reward shaping:** false negative here (missing a successful SSH brute-force) should carry the harshest penalty in your reward scheme, well beyond the standard -5 "service disrupted" — consider a distinct, larger penalty for "confirmed compromise missed" if you can encode that state in your simulation

---

## 9. Botnet ARES (5,508 rows — 0.92%)

**What it is:** Traffic from a host already infected with the ARES botnet malware, "phoning home" to its Command & Control (C2) server — periodic beacon traffic, receiving commands, or exfiltrating data.

**Detection fingerprint:** Periodic outbound connections at fixed/near-fixed intervals (heartbeat pattern) — very distinctive low-variance inter-arrival times to a small set of external destinations; often to unusual ports or via HTTP/DNS as C2 tunneling.

**Prevention:**
- Egress filtering — restrict what internal hosts can initiate outbound connections to
- DNS filtering / sinkholing known bad domains
- Endpoint protection (EDR) to prevent initial infection
- Network segmentation to limit lateral spread if one host is infected

**Incident response (manual):**
1. Identify the infected host from the beacon pattern
2. Isolate the host from the network immediately (contain before eradicating)
3. Run EDR/AV scan and malware removal; if severe, reimage the host
4. Block the C2 destination IP/domain network-wide
5. Investigate for lateral movement / other infected hosts communicating with the same C2

**AI agent behavior:**
- **Primary action:** `Isolate Server` (the infected host itself, not just the network path) — since the compromise lives on the endpoint, blocking the outbound IP alone leaves an infected machine on your network free to try other C2 channels
- **Secondary action:** `Block IP` for the specific C2 destination as a network-wide rule, in parallel with isolation
- **Note:** `Kill Process` is relevant here if your agent has endpoint-level integration (see Section 15 below) and can identify the specific malicious process generating the beacon
- **Reward shaping:** reward isolation strongly (+10) even though it's disruptive, because botnet presence is a standing risk (potential for DDoS participation, data exfiltration, lateral movement) that shouldn't be left running just to avoid short-term service disruption

---

## 10. Web Brute Force (2,733 rows — 0.46%)

**What it is:** Automated login-guessing against a web application's authentication endpoint (not a network-layer service like SSH/FTP, but an application-layer login form / API).

**Detection fingerprint:** High rate of POST requests to a login endpoint from one source, consistent request size (repeated form submission), short flow durations, high 401/403 response ratio if you have layer-7 visibility.

**Prevention:**
- Account lockout / exponential backoff after failed attempts
- CAPTCHA after N failures
- Rate limiting on the login endpoint specifically (separate from general site rate limits)
- MFA on the application

**Incident response (manual):**
1. Confirm via application logs whether any account was successfully compromised
2. Force password reset on targeted account(s) if compromise suspected
3. Block source IP at WAF/application layer
4. Enable/verify CAPTCHA and lockout policy going forward

**AI agent behavior:**
- **Primary action:** `Block IP`
- **Escalation:** `Revoke Credentials` if a successful login follows the brute-force burst (same logic as FTP/SSH-Patator)
- **Reward shaping:** similar to the other Patator-family attacks — this is fundamentally the same "credential brute force" *category* despite being application-layer rather than network-layer, so your Gymnasium environment can reuse the same reward function across FTP-Patator, SSH-Patator, and Web Brute Force with just a different feature distribution

---

## 11. Web XSS (1,357 rows — 0.23%)

**What it is:** Cross-Site Scripting — attacker injects malicious script into a web page (via input fields, URL parameters, etc.) that executes in *other users'* browsers, typically to steal session cookies or perform actions on their behalf. Note: XSS is fundamentally an application-layer/content attack, not a volumetric or connection-based one — network flow features are a weaker signal for this class than for DoS-style attacks.

**Detection fingerprint:** At the flow level, XSS payloads may show up as unusual request sizes/patterns to specific endpoints, but the *real* signal (script tags, encoded payloads) lives in the HTTP payload/body, which CICIDS2017's flow-level features don't directly capture — this is a known weak point for a pure flow-based autoencoder.

**Prevention:**
- Output encoding / context-aware escaping of all user-supplied content
- Content Security Policy (CSP) headers to restrict script execution sources
- Input validation and sanitization libraries (e.g., DOMPurify)
- WAF signatures for common XSS payload patterns
- HttpOnly + Secure cookie flags to limit session-theft impact even if XSS succeeds

**Incident response (manual):**
1. Identify and patch the vulnerable input/output point in the application
2. Invalidate sessions/cookies that may have been stolen via the injected script
3. Review logs for what data the injected script could have accessed or exfiltrated
4. Deploy/adjust CSP and WAF rule as a stopgap while the code fix is developed and deployed

**AI agent behavior:**
- **Primary action:** `Monitor` / log-and-alert with high priority for human review, rather than autonomous blocking — flow-level anomaly detection has limited precision for content-based attacks like XSS, so overconfident autonomous action (e.g., blocking a legitimate user IP) is a real false-positive risk
- **Recommendation:** for this class specifically, supplement your autoencoder+DQN pipeline with an application-layer signal (WAF logs / HTTP payload inspection) rather than relying on network flow features alone — worth noting explicitly as a **limitation** in your evaluation report
- **Reward shaping:** keep the false-alarm penalty relatively strict for autonomous `Block IP` on this class until precision is empirically validated on held-out data

---

## 12. Web SQL Injection (24 rows — 0.004%)

**What it is:** Attacker injects malicious SQL through application input fields to manipulate backend database queries — can read, modify, or delete data, or bypass authentication entirely.

**Detection fingerprint:** Same weak-signal problem as XSS — flow-level features poorly capture payload content. What little signal exists: unusual request sizes/timing to database-backed endpoints. With only 24 rows, **any performance metric on this class is not statistically meaningful** — flag this explicitly in your evaluation report, same caveat as Heartbleed.

**Prevention:**
- Parameterized queries / prepared statements (never string-concatenated SQL) — this is the single most effective control
- ORM usage with built-in escaping
- Least-privilege database accounts for the application (no `DROP`/`ALTER` rights for the web app's DB user)
- WAF SQLi signature rules as defense-in-depth (not a substitute for parameterized queries)

**Incident response (manual):**
1. Identify the vulnerable query/endpoint from WAF or application logs
2. Patch the code (parameterize the query) — this is a code fix, not a network control
3. Audit the database for signs of data exfiltration or tampering
4. Force credential rotation for any DB accounts potentially exposed

**AI agent behavior:**
- **Primary action:** `Monitor` and escalate to human/security team — with only 24 training samples this class cannot be reliably learned by your DQN; don't let the agent take disruptive autonomous action (`Isolate Server`) based on a class it has essentially no statistical basis to recognize confidently
- **Reward shaping:** exclude this class from quantitative DQN policy evaluation, or bucket it with XSS/generic "web attack, low confidence" and route to `Monitor` by default — document this explicitly as a known data limitation

---

## 13. Heartbleed (12 rows — 0.002%)

**What it is:** Exploits the OpenSSL Heartbeat extension vulnerability (CVE-2014-0160) — a malformed heartbeat request tricks a vulnerable OpenSSL version into leaking up to 64KB of process memory per request, potentially exposing private keys, session tokens, or credentials. Notably, this vulnerability itself was patched over a decade ago; its presence in CICIDS2017 is illustrative/historical.

**Detection fingerprint:** Small request to port 443 (TLS) followed by an unusually large response relative to the request — an inverted size ratio compared to normal TLS handshakes.

**Prevention:**
- Patch OpenSSL to a non-vulnerable version (this vulnerability has been fixed since 2014 — the real-world prevention is simply staying current on patching)
- Certificate/key rotation on any system that was ever exposed
- TLS library version monitoring as part of vulnerability management

**Incident response (manual):**
1. Patch OpenSSL immediately if an unpatched version is somehow still in use
2. Assume compromise of private keys/session data — rotate all TLS certificates and invalidate active sessions
3. Review logs for the small-request/large-response pattern to determine exposure window and volume of data leaked

**AI agent behavior:**
- **Primary action:** `Isolate Server` if detected — a successful Heartbleed exploit implies active key/memory leakage, which is severe enough to justify disruption despite the tiny sample size
- **Statistical caveat:** with only 12 rows, your DQN **cannot be meaningfully evaluated** on this class — document in your report that any precision/recall number here is not trustworthy, but the recommended action (isolate + patch) should still be hard-coded as a fallback rule rather than solely learned, given how severe (and rare) this attack is
- **Design suggestion:** for extremely rare, high-severity classes like this, consider a **hybrid approach**: a hardcoded rule ("port 443 + response/request size ratio > threshold → isolate + alert") that doesn't depend on the DQN having seen enough examples to learn a reliable policy

---

## General Design Guidance for Your Autonomous Agent

### What tools/integrations your agent needs (beyond the model files)

| Category | Tool / Integration | Used For |
|---|---|---|
| Network enforcement | Firewall API (iptables/nftables, cloud Security Groups, or a firewall vendor API) | Executing `Block IP` |
| Identity/Access | IAM API (Active Directory, LDAP, OAuth provider, or app-level user table) | Executing `Revoke Credentials` |
| Network segmentation | SDN controller or VLAN/switch API, or simply removing a host's routing/security-group access | Executing `Isolate Server` |
| Endpoint control | EDR agent API (e.g., a lightweight agent on hosts) | Executing `Kill Process` — this action requires *host-level* visibility your network-flow pipeline alone doesn't have; consider it a stretch goal or simulate it in Gymnasium for now |
| Logging/no-op | Your existing SQLAlchemy + PostgreSQL incident store | Executing `Monitor` — always log, even when no containment action is taken |
| Human-in-the-loop | Your WebSocket `/ws/alerts` channel + React dashboard | Escalation path when the agent's confidence is low or the action is `Isolate`/`Revoke` (irreversible/disruptive actions may warrant an "auto-execute vs. propose-and-confirm" mode toggle) |
| Reputation/context | A simple internal allowlist/reputation store (e.g., "is this IP a known internal scanner or CI system?") | Reducing false positives on Port Scan and brute-force classes |

### How the agent should decide, end to end

1. **State construction:** the DQN's state isn't just the raw 115-dim feature vector — it should include the autoencoder's reconstruction error (confidence signal), a rough attack-family hint if your architecture allows a lightweight classifier alongside the autoencoder, and contextual metadata (source reputation, asset criticality of the destination).
2. **Action selection:** `argmax Q(s, a)` over the 5 actions as your README already defines, but consider **action masking** — e.g., don't let the agent select `Revoke Credentials` for a state that has no associated identity/session context (Port Scan, DoS floods have no "credential" to revoke).
3. **Confidence gating:** for irreversible/high-impact actions (`Isolate Server`, `Revoke Credentials`), consider requiring the Q-value margin between the top action and runner-up to exceed a threshold before auto-executing; otherwise route to `Monitor` + human alert. This directly protects your false-positive rate on rare/ambiguous classes (XSS, SQLi, Heartbleed).
4. **Execution:** the selected action calls the relevant tool API above; result (success/failure) and any downstream telemetry (did the anomaly signal stop after action?) become the **reward signal** for online reinforcement (if you extend beyond pure Gymnasium simulation).
5. **Logging:** every decision — including `Monitor` — writes an incident record with the reconstruction error, chosen action, and rationale features, since this both feeds your PostgreSQL dashboard and gives you an audit trail for evaluating false positive/negative rates by attack class later.
6. **Escalation path:** treat DDoS (needs upstream provider), Web SQLi/XSS (weak flow-level signal), and Heartbleed (near-zero samples) as classes where the agent should default toward `Monitor` + escalate rather than aggressive autonomous action, and document this explicitly as a design decision in your evaluation report rather than a gap.

### Reward function cheat-sheet (extending your existing +10/-3/-5/+1 scheme)

- Correct containment of a confirmed, unambiguous attack (DoS floods, Port Scan, confirmed brute-force login): **+10**
- Correct `Monitor` on benign or low-confidence/rare-class traffic: **+1**
- Blocking/isolating on what turns out to be benign traffic: **-3** (false alarm)
- Taking a disruptive action that didn't stop the attack or broke a legitimate service: **-5**
- Missing (Monitor-only) a confirmed brute-force success or botnet beacon: consider a **distinct, larger penalty** than the standard -5, since these represent confirmed compromise rather than mere service disruption — your current 4-value reward scheme may be worth extending to 5 values to capture this distinction explicitly.

---

*Document prepared for: Real-Time Incident Tracking & Autonomous Threat Remediation, Jadavpur University — Dept. of IT*
*Aligned to milestone: Data Readiness & Engineering (15–21 June 2026), feeding into DQN environment design in AI Model Prototyping*