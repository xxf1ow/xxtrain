import json
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'deploy' / 'server'


class ServerTopologyTests(unittest.TestCase):
    def test_compose_bind_and_port_isolation(self):
        site = (ROOT / '.superpowers' / 'server-deployment' / 'test-site').resolve()
        if shutil.which('docker'):
            result = subprocess.run(
                ['docker', 'compose', '-f', str(DEPLOY / 'compose.yaml'), 'config', '--format', 'json'],
                env={**os.environ, 'XXTRAIN_SITE_ROOT': str(site)},
                capture_output=True,
                text=True,
                check=True,
            )
            config = json.loads(result.stdout)
        else:
            config = yaml.safe_load((DEPLOY / 'compose.yaml').read_text())
        self.assertEqual(config['name'], 'xxtrain-server')
        services = config['services']
        self.assertEqual(len(services), 24)
        for name, service in services.items():
            self.assertNotIn(service.get('restart'), ('always', 'unless-stopped'), name)
            for port in service.get('ports', []):
                host = port['host_ip'] if isinstance(port, dict) else port.split(':')[0]
                self.assertEqual(host, '127.0.0.1', (name, port))
            for mount in service.get('volumes', []):
                if isinstance(mount, dict):
                    if mount['type'] == 'tmpfs':
                        continue
                    self.assertEqual(mount['type'], 'bind')
                    source = mount['source']
                    target = mount['target']
                    readonly = mount.get('read_only', False)
                else:
                    source, target = mount.split(':/', 1)
                    target = '/' + target.split(':')[0]
                    readonly = mount.endswith(':ro')
                if source.startswith('${XXTRAIN_SITE_ROOT:?}/'):
                    source = str(site / source.removeprefix('${XXTRAIN_SITE_ROOT:?}/'))
                elif source.startswith('./'):
                    source = str((DEPLOY / source).resolve())
                self.assertTrue(Path(source).is_relative_to(site) or Path(source).is_relative_to(DEPLOY), (name, mount))
                if Path(source).is_relative_to(DEPLOY):
                    self.assertTrue(readonly, mount)
                if name == 'nginx' and target == '/etc/nginx/conf.d/default.conf':
                    self.assertEqual(Path(source), site / 'nginx.conf')
        self.assertEqual(
            Path(services['cvat_ui']['build']['context']).resolve()
            if shutil.which('docker')
            else (DEPLOY / services['cvat_ui']['build']['context']).resolve(),
            ROOT,
        )
        self.assertEqual(services['cvat_server']['image'], 'cvat/server:v2.51.0')
        self.assertEqual(services['clearml_apiserver']['image'], 'clearml/server:2.4.0')
        self.assertEqual(services['nginx']['image'], 'nginx:1.27-alpine')
        for name, published in [('cvat_server', '18080'), ('clearml_apiserver', '18083')]:
            self.assertIn(published, str(services[name]['ports']))
        for name, target in [
            ('cvat_redis_ondisk', '/var/lib/kvrocks'),
            ('cvat_clickhouse', '/var/log/clickhouse-server'),
        ]:
            mounts = services[name]['volumes']
            targets = [
                mount['target'] if isinstance(mount, dict) else '/' + mount.split(':/', 1)[1].split(':')[0]
                for mount in mounts
            ]
            self.assertIn(target, targets)

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

    def test_websocket_proxy_uses_http_1_1(self):
        template = (DEPLOY / 'nginx.conf').read_text()
        for path in ('/api/', '/'):
            body = re.search(r'location ' + re.escape(path) + r'\s*\{([^}]+)\}', template).group(1)
            self.assertIn('proxy_set_header Upgrade $http_upgrade;', body)
            self.assertIn('proxy_http_version 1.1;', body)


if __name__ == '__main__':
    unittest.main()
