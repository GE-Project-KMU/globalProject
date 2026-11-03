"""Forward-only, service-owned schema migrations; SQLite and PostgreSQL.

The JSONB aggregate remains canonical. Inspection's optional PostGIS projection
is generated from that same row, so deleting evidence deletes its geometry too.
"""
import os
from sqlalchemy import text

def migrate(engine, service, metadata):
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": "roadwatch-migrations-"+service})
        conn.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, description VARCHAR(200) NOT NULL, applied_at VARCHAR(40) NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
        versions=set(conn.execute(text("SELECT version FROM schema_migrations")).scalars())
        if 1 not in versions:
            metadata.create_all(conn)
            conn.execute(text("INSERT INTO schema_migrations(version,description) VALUES (1,'Service-owned canonical aggregates')"))
        if 2 not in versions:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_aggregates_org_kind_created ON aggregates(org,kind,created)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_aggregates_org_kind_expires ON aggregates(org,kind,expires)"))
            conn.execute(text("INSERT INTO schema_migrations(version,description) VALUES (2,'Tenant collection and retention indexes')"))
        if service == "inspection" and engine.dialect.name == "postgresql" and os.getenv("ROAD_POSTGIS_ENABLED") == "1" and 3 not in versions:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.execute(text("""ALTER TABLE aggregates ADD COLUMN IF NOT EXISTS observation_point geography(Point,4326)
                GENERATED ALWAYS AS (CASE WHEN kind='observation'
                AND payload->'payload'->'location'->>'lat' IS NOT NULL
                AND payload->'payload'->'location'->>'lon' IS NOT NULL
                THEN ST_SetSRID(ST_MakePoint((payload->'payload'->'location'->>'lon')::double precision,
                (payload->'payload'->'location'->>'lat')::double precision),4326)::geography ELSE NULL END) STORED"""))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_inspection_observation_point ON aggregates USING GIST(observation_point) WHERE kind='observation'"))
            conn.execute(text("INSERT INTO schema_migrations(version,description) VALUES (3,'Inspection PostGIS generated vehicle position and GiST index')"))
        if service == "inspection" and engine.dialect.name == "postgresql" and os.getenv("ROAD_POSTGIS_ENABLED") == "1" and 4 not in versions:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_inspection_observation_bbox ON aggregates USING GIST((observation_point::geometry)) WHERE kind='observation'"))
            conn.execute(text("INSERT INTO schema_migrations(version,description) VALUES (4,'Geometry rectangle index for inclusive longitude/latitude bbox semantics')"))

def main():
    import argparse
    parser=argparse.ArgumentParser(description="Apply migrations for one service's own database")
    parser.add_argument("service",choices=["identity","media","inspection","analysis","reporting"])
    args=parser.parse_args()
    from common.runtime import Store
    store=Store(args.service)
    with store.engine.connect() as conn:
        print({"service":args.service,"dialect":store.engine.dialect.name,"versions":list(conn.execute(text("SELECT version FROM schema_migrations ORDER BY version")).scalars())})

if __name__=="__main__":main()
