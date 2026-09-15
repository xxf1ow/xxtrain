FROM cvat/ui:v2.51.0

USER root
COPY src/xxtrain/integrations/cvat/ui/return-plugin.js /usr/share/nginx/html/xxtrain/return-plugin.js
RUN set -eu; \
    index=/usr/share/nginx/html/index.html; \
    test "$(grep -o '<head>' "$index" | wc -l)" -eq 1; \
    test "$(grep -o '</head>' "$index" | wc -l)" -eq 1; \
    sed -i 's#<head>#<head><style id="xxtrain-cvat-shell">.cvat-header,.cvat-annotation-header-menu-button{display:none!important}</style>#' "$index"; \
    sed -i 's#</head>#<script defer src="/xxtrain/return-plugin.js"></script></head>#' "$index"; \
    grep -Fq 'id="xxtrain-cvat-shell"' "$index"; \
    grep -Fq 'src="/xxtrain/return-plugin.js"' "$index"
USER 101

EXPOSE 8000
