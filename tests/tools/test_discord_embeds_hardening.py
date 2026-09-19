import json, pytest
from tools.discord_api.embeds import *

@pytest.mark.parametrize("kwargs",[
    {"title":b"x"},{"title":["x"]},{"description":1},
])
def test_embed_text_types_rejected(kwargs):
    with pytest.raises(EmbedValidationError): Embed(**kwargs)
@pytest.mark.parametrize("ctor,args",[
    (EmbedField,(b"n","v")),(EmbedField,("n",b"v")),(EmbedAuthor,(b"a",)),(EmbedFooter,(b"f",)),
])
def test_component_text_types_rejected(ctor,args):
    with pytest.raises(EmbedValidationError): ctor(*args)
@pytest.mark.parametrize("value",[1,"yes",None])
def test_inline_requires_bool(value):
    with pytest.raises(EmbedValidationError): EmbedField("n","v",inline=value)
def test_color_bool_rejected():
    with pytest.raises(EmbedValidationError): Embed(color=True)
@pytest.mark.parametrize("kwargs",[{"author":"x"},{"footer":"x"},{"fields":["x"]},{"fields":1}])
def test_nested_types_rejected(kwargs):
    with pytest.raises(EmbedValidationError): Embed(**kwargs)
def test_fields_runtime_is_tuple_and_annotation_is_immutable_surface():
    e=Embed(fields=[EmbedField("n","v")]); assert isinstance(e.fields,tuple); assert "Sequence" in str(Embed.__annotations__["fields"])
@pytest.mark.parametrize("url",[
    "https://example.com\n","https://exa\tmple.com","https://exa mple.com","https://example.com\r",
    "https://[","https://example.com:bad","https://example.com:70000",
    "https://\u00a0example.com","https://example.com\\@evil.com","https://%zz.example.com","https://example.com/%zz",
])
def test_bad_urls_normalized_to_embed_error(url):
    with pytest.raises(EmbedValidationError): Embed(url=url)
def test_http_urls_with_valid_port_path_ok(): Embed(url="https://example.com:443/a?b=c#d")
@pytest.mark.parametrize("field",["image_url","thumbnail_url"])
def test_attachment_media_allowed(field): Embed(**{field:"attachment://image.png"})
def test_attachment_icons_allowed():
    Embed(author=EmbedAuthor("a",icon_url="attachment://a.png"),footer=EmbedFooter("f",icon_url="attachment://f.png"))
@pytest.mark.parametrize("value",["attachment://","attachment://x/y","attachment://x?y=1","attachment://x#frag","attachment://x y"])
def test_invalid_attachment_rejected(value):
    with pytest.raises(EmbedValidationError): Embed(image_url=value)
def test_attachment_not_allowed_for_clickthrough_url():
    with pytest.raises(EmbedValidationError): Embed(url="attachment://x")
def test_role_mention_detected(): assert contains_mention("<@&123456>")
def test_channel_reference_not_treated_as_ping(): assert not contains_mention("<#123456>")
def test_contains_mention_requires_string():
    with pytest.raises(EmbedValidationError): contains_mention(None)
def test_validate_embeds_bad_members():
    with pytest.raises(EmbedValidationError): validate_embeds([Embed(),"bad"])
    with pytest.raises(EmbedValidationError): validate_embeds(None)
def test_every_accepted_payload_json_serializes():
    samples=[Embed(),Embed(title="T",description="D",color=0,fields=[EmbedField("N","V",True)]),Embed(image_url="attachment://x.png"),Embed(author=EmbedAuthor("A",url="https://example.com"))]
    for e in samples: json.dumps(e.to_payload())
def test_fallback_includes_nontext_references_and_linked_title():
    e=Embed(author=EmbedAuthor("A",url="https://a.example"),title="T",url="https://t.example",timestamp="2026-08-14T10:00:00Z",image_url="https://i.example/x.png",thumbnail_url="https://i.example/t.png")
    text=embed_to_plain_text(e)
    for expected in ["[A](https://a.example)","[T](https://t.example)","2026-08-14T10:00:00Z","https://i.example/x.png","https://i.example/t.png"]: assert expected in text
def test_fallback_requires_embed():
    with pytest.raises(EmbedValidationError): embed_to_plain_text("bad")
@pytest.mark.parametrize("ts",["2026-08-14T25:00:00Z","2026-08-14T10:00:00+24:00","2026-02-30T00:00:00Z"])
def test_semantically_invalid_timestamps_rejected(ts):
    with pytest.raises(EmbedValidationError): Embed(timestamp=ts)

def test_randomized_typed_boundary_never_leaks_or_emits_non_json_payloads():
    import random
    import string

    rng = random.Random(86324)
    alphabet = string.ascii_letters + string.digits + ":/?#[]@!$&'()*+,;=%\\ \t\r\n\x00\u00a0"
    scalar_pool = [None, True, False, 0, 1, -1, 0xFFFFFF, 0x1000000, b"bytes", ["list"]]

    for _ in range(5000):
        candidate = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 80)))
        kwargs = {
            "title": rng.choice([candidate, *scalar_pool]),
            "description": rng.choice([candidate, *scalar_pool]),
            "url": rng.choice([candidate, None]),
            "image_url": rng.choice([candidate, None]),
            "thumbnail_url": rng.choice([candidate, None]),
            "color": rng.choice(scalar_pool),
        }
        try:
            embed = Embed(**kwargs)
        except EmbedValidationError:
            continue
        json.dumps(embed.to_payload())

    for _ in range(5000):
        candidate = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 120)))
        try:
            embed = Embed(url=candidate)
        except EmbedValidationError:
            continue
        json.dumps(embed.to_payload())


# ── Additional adversarial closure (F1-F6) ─────────────────────────────────
@pytest.mark.parametrize("url",[
    "https://example.com/x%0ay","https://example.com/x%0d%0ay","https://example.com/x%09y",
    "https://example.com/x%00y","https://example.com/x%20y","https://example.com/x%5cy",
    "https://evil%0a.com",
])
def test_percent_encoded_forbidden_chars_rejected(url):
    with pytest.raises(EmbedValidationError): Embed(url=url)

@pytest.mark.parametrize("url",[
    "https://user:pass@evil.com/","https://evil.com@good.com/","https://***@evil.com",
    "https://@evil.com","https://user@evil.com",
])
def test_userinfo_credentials_rejected(url):
    with pytest.raises(EmbedValidationError): Embed(url=url)

def test_bidi_format_chars_rejected():
    with pytest.raises(EmbedValidationError): Embed(url="https://‮example.com")
    with pytest.raises(EmbedValidationError): Embed(url="https://example.com/​")

def test_garbage_after_ipv6_literal_rejected():
    with pytest.raises(EmbedValidationError): Embed(url="http://[::1]x")

def test_valid_ipv6_authority_accepted():
    Embed(url="https://[::1]/")
    Embed(url="https://[::1]:8443/")

@pytest.mark.parametrize("url",["attachment://..","attachment://a%2fb","attachment://a/b"])
def test_attachment_nonfilename_rejected(url):
    with pytest.raises(EmbedValidationError): Embed(image_url=url)

def test_attachment_valid_filenames_accepted():
    Embed(image_url="attachment://x.png")
    Embed(image_url="attachment://a_b-1")

def test_timestamp_out_of_range_offset_rejected():
    with pytest.raises(EmbedValidationError): Embed(timestamp="2026-08-14T10:00:00+12:60")

def test_timestamp_valid_offset_accepted():
    Embed(timestamp="2026-08-14T10:00:00+05:30")
