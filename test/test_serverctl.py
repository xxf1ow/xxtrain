import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'deploy' / 'server'


class ServerTopologyTests(unittest.TestCase):
    def test_compose_bind_and_port_isolation(self):
        config = yaml.safe_load((DEPLOY / 'compose.yaml').read_text())
        self.assertEqual(config['name'], 'xxtrain-server')
        services = config['services']
        self.assertEqual(len(services), 24)
        for name, service in services.items():
            self.assertNotIn(service.get('restart'), ('always', 'unless-stopped'), name)
            for port in service.get('ports', []):
                self.assertTrue(port.startswith('127.0.0.1:'), (name, port))
            for mount in service.get('volumes', []):
                if isinstance(mount, dict):
                    self.assertEqual(mount['type'], 'tmpfs')
                    continue
                source = mount.split(':/')[0]
                self.assertTrue(source.startswith('${XXTRAIN_SITE_ROOT:?}/') or source.startswith('./'), (name, mount))
                if source.startswith('./'):
                    self.assertTrue(mount.endswith(':ro'), mount)
        self.assertIn(
            '${XXTRAIN_SITE_ROOT:?}/nginx.conf:/etc/nginx/conf.d/default.conf:ro', services['nginx']['volumes']
        )
        self.assertIn('${XXTRAIN_SITE_ROOT:?}/cvat/kvrocks:/var/lib/kvrocks', services['cvat_redis_ondisk']['volumes'])
        self.assertIn(
            '${XXTRAIN_SITE_ROOT:?}/cvat/clickhouse-logs:/var/log/clickhouse-server',
            services['cvat_clickhouse']['volumes'],
        )
        self.assertEqual(services['cvat_ui']['build']['context'], '../..')
        self.assertEqual(services['cvat_server']['image'], 'cvat/server:v2.51.0')
        self.assertEqual(services['clearml_apiserver']['image'], 'clearml/server:2.4.0')
        self.assertEqual(services['nginx']['image'], 'nginx:1.27-alpine')
        self.assertEqual(services['cvat_server']['ports'], ['127.0.0.1:18080:8080'])
        self.assertEqual(services['clearml_apiserver']['ports'], ['127.0.0.1:18083:8008'])

    def test_nginx_session_gate_and_listeners(self):
        template = (DEPLOY / 'nginx.conf').read_text()
        self.assertNotIn('192.168.0.109', template)
        self.assertEqual(len(re.findall(r'listen \{\{LAN_ADDRESS\}\}:\d+;', template)), 4)
        session = re.search(r'location = /_cvat_session\s*\{([^}]+)\}', template).group(1)
        self.assertIn('internal;', session)
        self.assertIn('proxy_pass http://127.0.0.1:18001/platform/api/session;', session)
        self.assertIn('proxy_set_header Cookie $http_cookie;', session)
        for path in ('/assets/', '/api/', '/static/', '/'):
            body = re.search(r'location ' + re.escape(path) + r'\s*\{([^}]+)\}', template).group(1)
            self.assertIn('auth_request /_cvat_session;', body, path)
        platform = re.search(r'location /platform/\s*\{([^}]+)\}', template).group(1)
        self.assertNotIn('auth_request', platform)


if __name__ == '__main__':
    unittest.main()
