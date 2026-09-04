"""Fixed Allegro-xArm6 contact vocabulary used by dexCG."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactLink:
    token_name: str
    dexart_link: str
    source_shadow_token: str

    @property
    def token(self) -> str:
        return f"<{self.token_name}>"


# This is a permanent model contract. link_13.0 is a thumb-base segmentation
# link in DexArt, not one of its contact/imagination links.
ALLEGRO_CONTACT_LINKS = (
    ContactLink("allegro_palm", "base_link", "rh_palm"),
    ContactLink("allegro_thumb_tip", "link_15.0_tip", "rh_thdistal"),
    ContactLink("allegro_thumb_middle", "link_15.0", "rh_thmiddle"),
    ContactLink("allegro_thumb_proximal", "link_14.0", "rh_thproximal"),
    ContactLink("allegro_index_tip", "link_3.0_tip", "rh_ffdistal"),
    ContactLink("allegro_index_distal", "link_3.0", "rh_ffmiddle"),
    ContactLink("allegro_index_middle", "link_2.0", "rh_ffproximal"),
    ContactLink("allegro_index_proximal", "link_1.0", "rh_ffknuckle"),
    ContactLink("allegro_middle_tip", "link_7.0_tip", "rh_mfdistal"),
    ContactLink("allegro_middle_distal", "link_7.0", "rh_mfmiddle"),
    ContactLink("allegro_middle_middle", "link_6.0", "rh_mfproximal"),
    ContactLink("allegro_middle_proximal", "link_5.0", "rh_mfknuckle"),
    ContactLink("allegro_ring_tip", "link_11.0_tip", "rh_rfdistal"),
    ContactLink("allegro_ring_distal", "link_11.0", "rh_rfmiddle"),
    ContactLink("allegro_ring_middle", "link_10.0", "rh_rfproximal"),
    ContactLink("allegro_ring_proximal", "link_9.0", "rh_rfknuckle"),
)

ALLEGRO_CONTACT_TOKENS = tuple(link.token for link in ALLEGRO_CONTACT_LINKS)
SHADOW_TO_ALLEGRO_TOKEN = {
    f"<{link.source_shadow_token}>": link.token for link in ALLEGRO_CONTACT_LINKS
}


def token_to_dexart_link() -> dict[str, str]:
    return {link.token: link.dexart_link for link in ALLEGRO_CONTACT_LINKS}
