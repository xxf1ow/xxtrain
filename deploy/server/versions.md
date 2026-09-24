# Server container versions

The single Compose manifest is [compose.yaml](compose.yaml). It translates the official [CVAT Compose at `v2.51.0` (`dfd505e7622de2e4aed214ce3119d12bf2f7a22a`)](https://github.com/cvat-ai/cvat/blob/dfd505e7622de2e4aed214ce3119d12bf2f7a22a/docker-compose.yml), its [Vector config](https://github.com/cvat-ai/cvat/blob/dfd505e7622de2e4aed214ce3119d12bf2f7a22a/components/analytics/vector/vector.toml), and the [ClearML Server Compose at `v2.4.0` (`0a90a2745885fd7173cdb1d5466855fc57b1187b`)](https://github.com/clearml/clearml-server/blob/0a90a2745885fd7173cdb1d5466855fc57b1187b/docker/docker-compose.yml). The ClearML upstream manifest uses mutable `clearml/server:latest`; this deployment selects its published `clearml/server:2.4.0` tag. These are version tags, not content digests; the publisher can replace a tag.

| Services | Image reference |
| --- | --- |
| CVAT server and eight workers | `cvat/server:v2.51.0` |
| CVAT UI base for `xxtrain-cvat-ui:2.51.0` | `cvat/ui:v2.51.0` |
| CVAT Postgres, Redis, Kvrocks | `postgres:15-alpine`, `redis:7.2.11-alpine`, `apache/kvrocks:2.12.1` |
| CVAT policy and analytics | `openpolicyagent/opa:0.63.0`, `clickhouse/clickhouse-server:23.11-alpine`, `timberio/vector:0.26.0-alpine` |
| ClearML API, Web, files and async deletion | `clearml/server:2.4.0` |
| ClearML MongoDB, Redis, Elasticsearch | `mongo:8.0.15`, `redis:8.2.3`, `elasticsearch:8.19.9` |
| Public reverse proxy | `nginx:1.27-alpine` |

CVAT Grafana and Traefik are omitted: nginx owns public routes, and the deployment does not expose upstream Grafana analytics. ClearML `agent-services` is omitted because this server deployment does not run an Agent. All application data uses `.deployment` bind mounts; the rendered nginx configuration, Vector's configuration, and the custom UI source are checked-in code assets.
