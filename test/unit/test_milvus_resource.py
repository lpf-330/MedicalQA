import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.resource_manager.milvus_connection.milvus_connection_resource import (
    MilvusConnectionResource,
    MilvusConnectionConfig,
    MilvusConnectionFactory,
    MilvusConnectionClient,
)


class TestMilvusConnectionResource:

    @patch('src.resource_manager.milvus_connection.milvus_connection_resource.MilvusAdapterImpl')
    def test_milvus_connection_resource_activate(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        MockAdapterImpl.return_value = mock_adapter

        config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        resource = MilvusConnectionResource(config)
        resource.activate()

        MockAdapterImpl.assert_called_once()
        mock_adapter.connect.assert_called_once_with(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        assert resource.is_activate() is True
        assert resource.get_adapter() is mock_adapter

    @patch('src.resource_manager.milvus_connection.milvus_connection_resource.MilvusAdapterImpl')
    def test_milvus_connection_resource_deactivate(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        MockAdapterImpl.return_value = mock_adapter

        config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        resource = MilvusConnectionResource(config)
        resource.activate()
        resource.deactivate()

        mock_adapter.disconnect.assert_called_once()
        assert resource.is_activate() is False

    @patch('src.resource_manager.milvus_connection.milvus_connection_resource.MilvusAdapterImpl')
    def test_milvus_connection_resource_destroy(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        MockAdapterImpl.return_value = mock_adapter

        config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        resource = MilvusConnectionResource(config)
        resource.activate()
        resource.destroy()

        mock_adapter.disconnect.assert_called_once()
        assert resource.get_adapter() is None
        assert resource.is_activate() is False

    def test_milvus_connection_resource_get_type(self):
        config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        resource = MilvusConnectionResource(config)
        assert resource.get_type() == "milvus_connection"


class TestMilvusConnectionConfig:

    def test_milvus_connection_config_validate(self):
        valid_config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        assert valid_config.validate() is True

        invalid_config_uri = MilvusConnectionConfig(
            uri="",
            user="root",
            password="milvus",
            token=""
        )
        assert invalid_config_uri.validate() is False

        invalid_config_user = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="",
            password="milvus",
            token=""
        )
        assert invalid_config_user.validate() is False

        invalid_config_password = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="",
            token=""
        )
        assert invalid_config_password.validate() is False


class TestMilvusConnectionFactory:

    def test_milvus_connection_factory_create(self):
        config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        factory = MilvusConnectionFactory()
        resource = factory.create(config)

        assert isinstance(resource, MilvusConnectionResource)
        assert resource.get_type() == "milvus_connection"

    def test_milvus_connection_factory_create_wrong_config(self):
        wrong_config = MagicMock()
        factory = MilvusConnectionFactory()
        with pytest.raises(TypeError):
            factory.create(wrong_config)


class TestMilvusConnectionClient:

    @patch('src.resource_manager.milvus_connection.milvus_connection_resource.MilvusAdapterImpl')
    def test_milvus_connection_client_search(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        mock_adapter.search.return_value = [
            {"id": "entity_1", "distance": 0.85, "entity": {"name": "糖尿病"}}
        ]
        MockAdapterImpl.return_value = mock_adapter

        config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        resource = MilvusConnectionResource(config)
        resource.activate()

        client = MilvusConnectionClient(resource)
        results = client.search(
            collection_name="medical_entity",
            query_vector=[0.1] * 1024,
            top_k=5
        )

        mock_adapter.search.assert_called_once_with(
            collection_name="medical_entity",
            query_vector=[0.1] * 1024,
            top_k=5
        )
        assert len(results) == 1
        assert results[0]["id"] == "entity_1"

    @patch('src.resource_manager.milvus_connection.milvus_connection_resource.MilvusAdapterImpl')
    def test_milvus_connection_client_hybrid_search(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        mock_adapter.hybrid_search.return_value = [
            {"id": "entity_1", "score": 0.92, "name": "糖尿病", "neo4j_node_id": "disease_001"}
        ]
        MockAdapterImpl.return_value = mock_adapter

        config = MilvusConnectionConfig(
            uri="http://localhost:19530",
            user="root",
            password="milvus",
            token=""
        )
        resource = MilvusConnectionResource(config)
        resource.activate()

        client = MilvusConnectionClient(resource)
        results = client.hybrid_search(
            query_vector=[0.1] * 1024,
            collections=["medical_entity", "entity_relations"],
            top_k=3,
            weights={"medical_entity": 0.6, "entity_relations": 0.4}
        )

        mock_adapter.hybrid_search.assert_called_once_with(
            query_vector=[0.1] * 1024,
            collections=["medical_entity", "entity_relations"],
            top_k=3,
            weights={"medical_entity": 0.6, "entity_relations": 0.4}
        )
        assert len(results) == 1
