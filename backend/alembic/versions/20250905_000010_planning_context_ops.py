"""planning, context, and ops tables

Revision ID: 000010_planning_context_ops
Revises: 000009_documents_kpi
Create Date: 2025-09-05 00:36:00

"""
from alembic import op
import sqlalchemy as sa


revision = '000010_planning_context_ops'
down_revision = '000009_documents_kpi'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Planning core
    op.create_table(
        'samps',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('period_start', sa.String(), nullable=True),
        sa.Column('period_end', sa.String(), nullable=True),
        sa.Column('value_definition', sa.String(), nullable=True),
        sa.Column('decision_criteria', sa.String(), nullable=True),
        sa.Column('risk_appetite', sa.String(), nullable=True),
        sa.Column('finance_link', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'am_objectives',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('samp_id', sa.String(), nullable=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('measure', sa.String(), nullable=True),
        sa.Column('target', sa.String(), nullable=True),
        sa.Column('due_date', sa.String(), nullable=True),
        sa.Column('owner_id', sa.String(), nullable=True),
        sa.Column('stakeholder_ref', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'am_plans',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('objective_id', sa.String(), nullable=True),
        sa.Column('resources_json', sa.Text(), nullable=True),
        sa.Column('schedule_json', sa.Text(), nullable=True),
        sa.Column('lifecycle_method', sa.String(), nullable=True),
        sa.Column('eval_method', sa.String(), nullable=True),
        sa.Column('cost_impact', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'risks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('asset_id', sa.String(), nullable=True),
        sa.Column('process_ref', sa.String(), nullable=True),
        sa.Column('cause', sa.String(), nullable=True),
        sa.Column('likelihood', sa.String(), nullable=True),
        sa.Column('consequence', sa.String(), nullable=True),
        sa.Column('rating', sa.String(), nullable=True),
        sa.Column('treatment_plan', sa.String(), nullable=True),
        sa.Column('owner_id', sa.String(), nullable=True),
        sa.Column('due_date', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'opportunities',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('expected_value', sa.String(), nullable=True),
        sa.Column('plan', sa.String(), nullable=True),
        sa.Column('owner_id', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'context_issues',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('climate_relevance', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'stakeholders',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=True),
        sa.Column('requirements_json', sa.Text(), nullable=True),
        sa.Column('comm_prefs', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'policies',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('approver_id', sa.String(), nullable=True),
        sa.Column('s3_key', sa.String(), nullable=True),
        sa.Column('effective_date', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'role_map',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('responsibilities', sa.String(), nullable=True),
        sa.Column('authority', sa.String(), nullable=True),
        sa.Column('owner_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'leadership_attestations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('statement', sa.String(), nullable=True),
        sa.Column('date', sa.String(), nullable=True),
        sa.Column('signer_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'assets',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('parent_id', sa.String(), nullable=True),
        sa.Column('tag', sa.String(), nullable=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=True),
        sa.Column('criticality', sa.String(), nullable=True),
        sa.Column('condition', sa.String(), nullable=True),
        sa.Column('performance_json', sa.Text(), nullable=True),
        sa.Column('lifecycle_state', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'work_orders',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('asset_id', sa.String(), nullable=True),
        sa.Column('type', sa.String(), nullable=True),
        sa.Column('priority', sa.String(), nullable=True),
        sa.Column('sla_due', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'providers',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('scope', sa.String(), nullable=True),
        sa.Column('risks_json', sa.Text(), nullable=True),
        sa.Column('monitoring_criteria', sa.String(), nullable=True),
        sa.Column('performance_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    op.create_table(
        'improvements',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('source_ref', sa.String(), nullable=True),
        sa.Column('priority', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )


def downgrade() -> None:
    for table in [
        'improvements','providers','work_orders','assets','leadership_attestations','role_map','policies',
        'stakeholders','context_issues','opportunities','risks','am_plans','am_objectives','samps']:
        try:
            op.drop_table(table)
        except Exception:
            pass

