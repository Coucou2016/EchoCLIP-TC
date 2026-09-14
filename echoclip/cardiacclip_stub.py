"""Backward-compatible alias for ``echoclip.cardiacclip``.

Prefer ``from echoclip.cardiacclip import ...``.
"""

from echoclip.cardiacclip import (  # noqa: F401
    CARDIACCLIP_DOWNLOAD,
    CARDIACCLIP_REQUIRES,
    CardiacCLIPCompareResult,
    assert_paper_ready,
    comparison_table_row,
    document_download,
    document_gaps,
    load_cardiacclip_interface,
    resolve_weights_path,
    write_comparison_template,
)
