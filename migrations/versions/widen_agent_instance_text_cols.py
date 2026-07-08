"""widen user_agent_instances.instance_name/display_name to Text

Revision ID: widen_agent_instance_text_cols
Revises: fix_public_user_uuid_storage
Create Date: 2026-07-08

Privacy Faza 2 follow-up. The ORM model widened ``instance_name``
(``String(100)`` -> ``Text``) and ``display_name`` (``String(200)`` -> ``Text``)
as a preventive measure. Both columns stay PLAINTEXT (``instance_name`` is a SQL
lookup key; ``display_name`` is a UI label) — the encrypted fields
(``description``/``system_prompt_*``) were already ``Text``/JSON, so no schema
change is needed there. This migration brings existing databases in line so the
schema-as-code contract holds and ``alembic revision --autogenerate`` reports no
drift.

Batch mode is used so the change is portable to SQLite (which cannot
``ALTER COLUMN ... TYPE`` in place and recreates the table); the recreate
reflects and preserves the existing ``uq_user_agent_instance_user_name`` unique
constraint and the ``instance_name`` index. On PostgreSQL/MySQL the change is an
in-place, lossless widening.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "widen_agent_instance_text_cols"
down_revision: Union[str, Sequence[str], None] = "fix_public_user_uuid_storage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Widen the two plaintext name columns to Text."""
    with op.batch_alter_table("user_agent_instances") as batch_op:
        batch_op.alter_column(
            "instance_name",
            existing_type=sa.String(length=100),
            type_=sa.Text(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "display_name",
            existing_type=sa.String(length=200),
            type_=sa.Text(),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Narrow the columns back to their original String widths.

    Data written while the columns were ``Text`` may exceed the original bounds;
    this reverses the type only and does not truncate (PostgreSQL/MySQL will
    reject over-long values at ``ALTER`` time, which is the correct fail-loud
    signal that a downgrade would lose data).
    """
    with op.batch_alter_table("user_agent_instances") as batch_op:
        batch_op.alter_column(
            "display_name",
            existing_type=sa.Text(),
            type_=sa.String(length=200),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "instance_name",
            existing_type=sa.Text(),
            type_=sa.String(length=100),
            existing_nullable=False,
        )
