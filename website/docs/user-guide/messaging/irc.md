# irc for hermes-gateway

An IRC client for hermes with multilines and very long lines support (`draft/multiline`),
enabling multiline markdown messages (with possible colorization in clients).

Warning: IRC is a protocol which can not be considered secure at all. 
Even if you use it with TLS, admins can still see everything you send, 
and potentially impersonate you or other nicknames.

For these reasons, we recommend you use another platform with end-to-end encryption,
or at like that host your own private IRC server, if you do:
* Consider using a modern irc server like [Ergo](https://ergo.chat/) (written in go),
* Enable draft/multiline to allow markdown (for Ergo, it's enabled by default)
* Enable TLS for all client connections if possible
* Restrict server access via:
  - Ideally port forwarding (so it's all private)
  - Or IRC server password and firewalls
  - Client certificate authentication

Despite these strong concerns with IRC, we still provide it because:

- Other platforms are often proprietary (feishu, telegram, signal), 
- or heavy/complex (e.g. matrix).
- the simplicity of IRC can sometimes be see as an advantage in certain situation.

### Configuration

IRC bots get configured by environment variables.

The location is in `~/.hermes/profiles/<name>/.env` (or in ~/.hermes/.env for the main profile).

```bash
IRC_SERVER=irc.example.com
IRC_PORT=6667            # or 6697 if IRC_USE_TLS
IRC_USE_TLS=false        # true for port 6697
IRC_PASSWORD=            # optional password to connect to the server

IRC_NICK=MyBotNick
IRC_NICKSERV_PASSWORD=xyz      # optional, it will use /privmsg NickServ identify <IRC_NICK> <IRC_NICKSERV_PASSWORD>
IRC_NICKSERV_SERVICE=NickServ
IRC_USERNAME=mybot       # optional, defaults to IRC_NICK
IRC_REALNAME="My AI Bot" # optional, defaults to "Hermes Agent"
IRC_CHANNELS=#channel1,#channel2  # comma-separated list of channels to join
IRC_ALLOWED_USERS=alice,bob    # case insensitive nicknames, use * to allow all users to talk to the bot

IRC_MESSAGE_CHUNK_LIMIT=16384  # classically, irc chunks are 350 chars, but for multiline it's best to use more.
IRC_REQUIRE_MULTILINE=true     # if possible use a server which has +draft/multiline
```

### Channel mentions

For channels only, the bot only responds to messages that mention it in the list of nicknames at the very beginning,
provided the sender nick is in `IRC_ALLOWED_USERS`.

- Example: `Foo: hello` -> only agent 'Foo' would answer.
- Example: `Foo: Bar: Baz: hello` -> the three agents 'Foo', 'Bar', 'Baz' would all answer.

- **DMs are not affected** — they work normally

### IRCv3 Multiline Support

The IRC adapter supports IRCv3 `draft/multiline` capability 
for sending and receiving multiline messages as a single unit. This allows 
markdown to be sent and received! You will however need a client
that supports this, and a plugin to render markdown.

How it works:

1. **CAP negotiation on connect:**
   - Sends `CAP LS 302` to request server capabilities
   - If `draft/multiline` is available, sends `CAP REQ :draft/multiline`
   - Handles `CAP ACK` / `CAP NAK` responses
   - Timeout: 5 seconds, then proceeds without multiline support

2. **Sending multiline messages:**
   - When `multiline_cap` is enabled and content contains newlines
   - Sends `BATCH +<id> draft/multiline <target>`
   - Each line tagged with `@batch=<id>`
   - Closes with `BATCH -<id>`
   - Blank lines preserved as single spaces for paragraph breaks
   - Falls back to separate PRIVMSG if capability not available

3. **Receiving multiline messages:**
   - Parses IRCv3 message tags (`@key=value` prefix)
   - Tracks incoming `draft/multiline` batches by ID
   - Buffers messages until `BATCH -<id>` received
   - Combines with `\n` separator preserving original structure
   - Emits single MessageEvent with combined text

Example

**Without draft/multiline:**
```
>>> PRIVMSG #channel :Hello
>>> PRIVMSG #channel :World
```

**With draft/multiline:**
```
>>> BATCH +m1 draft/multiline #channel
>>> @batch=m1 PRIVMSG #channel :Hello
>>> @batch=m1 PRIVMSG #channel :World
>>> BATCH -m1
```

The recipient sees the same result, but the messages are grouped as a single unit.

## Testing

**Channel mention-only mode:**

For channels only, the bot only responds to messages that mention it at the very beginning:

- Must start with `<nickname>:` (colon immediately after nickname, no space)
- Example: `YourBotName: hello` ✓ (responds), `hello YourBotName:` ✗ (ignored)
- **DMs are not affected** — they work normally
- Uses original `_nick` for detection (what users will type) while `_actual_nick` tracks what the server actually accepted
- This handles nick collisions correctly — if bot connects as `YourBotName_` due to name conflict, it still responds to `YourBotName:` mentions

**Quick development workflow (recommended for iterative changes):**

Use the multi-terminal skill (see `/skills devops/multi-terminal`) to run a 4-pane tmux session:

1. **Create session:**
   ```
   /multi-terminal start dev --project-dir ~/hermes-fork
   ```

2. **Start gateway:**
   ```
   /multi-terminal send dev gateway "source venv/bin/activate && python -m gateway.run"
   ```

3. **Connect test client:**
   ```
   /multi-terminal send dev irc "telnet 127.0.0.1 6667"
   # Then send IRC commands:
   # NICK TestUser
   # USER TestUser 0 * :TestUser
   # JOIN #home
   # PRIVMSG #home :YourBotName: hello
   ```

4. **Monitor logs:**
   ```
   /multi-terminal capture dev gateway 50
   /multi-terminal capture dev irc 20
   ```

5. **Edit files:**
   ```
   /multi-terminal send dev edit "vim gateway/platforms/irc.py"
   ```

6. **Restart gateway after changes:**
   ```
   tmux send-keys -t hermes-dev:0.1 C-c
   /multi-terminal send dev gateway "source venv/bin/activate && python -m gateway.run"
   ```

This workflow lets you simultaneously:
- Run the gateway with live logs
- Send test IRC messages
- Edit adapter code
- Iterate quickly without terminal switching


# Not implemented

## Mask verification

You may have wanted to have a `<nick>!<username>@<host>` verification (usually called a mask) 
instead of a simple nick list. However masks are not implemented right now.

The rationale behind this choice:

- Modern IRC networks reject the /nick command instantly when the target is registered and services enforce protection, eliminating the historical grace period entirely. 
- For pure identity verification, hostmask checking (<nick>!<user>@<host>) offers negligible benefit over verifying a registered, authenticated nick (Hostmasks are ephemeral (DHCP, mobile networks, bouncers), trivially spoofed (identd can be faked, proxies/VPNs mask hosts), and provide no cryptographic proof of identity.
- Therefore the only thing that could be useful is the `host`, when used with a cloak set by the server. But it won't arguably provide any additional security compared to a registered nickname.

# Todo

* [x] IRCv3 draft/multiline support
* [x] Message length splitting (~350 chars) for long messages when BATCH unavailable
* [x] Handle nick collision (433/436 errors) with fallback nick recovery
* [x] Add NickServ authentication support for networks requiring it

