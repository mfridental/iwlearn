# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt
import pyodbc
import traceback
from threading import current_thread
from couchbase.bucket import Bucket

from pymongo import MongoClient
from pyclickhouse import Connection

from base import BaseDataSource


class ThreadSafeDataSource(BaseDataSource):
    """
        Base for all data sources that must support data retrieval from multiple threads.

        You have to define a new client, like so
        self.create_client: lambda connection_string: YourClient(connection_string)

        if your client needs to close the connection define:
        self.close_client: lambda yourClient: yourClient.close()
    """
    clients = dict()
    MAX_THREADS = 100

    def __init__(self):
        BaseDataSource.__init__(self)
        self.create_client = None
        self.close_client = None

    def isin(self, a_dict, list_of_keys):
        for key in list_of_keys:
            if key not in a_dict:
                return False
        return True

    def delete_client(self, connection_string):
        thread_id = current_thread().ident

        logging.debug('delete_client %s' % (thread_id,))
        if thread_id in self.clients:
            logging.debug('delete_client %s' % (connection_string,))
            if connection_string in self.clients[thread_id]:
                try:
                    if self.close_client is not None:
                        self.close_client(self.clients[thread_id][connection_string])
                except:
                    logging.warning('Cannot close client (already closed?) %s' % traceback.format_exc())
                del self.clients[thread_id][connection_string]
                logging.debug('delete_client #clients: %s' % (len(self.clients[thread_id]),))

    def get_client(self, connection_string):
        thread_id = current_thread().ident

        if thread_id not in self.clients:
            if len(self.clients) < self.__class__.MAX_THREADS:
                self.clients[thread_id] = dict()
            else:
                raise Exception('%r tries to open too many connections' % self)
        if connection_string not in self.clients[thread_id]:
            self.clients[thread_id][connection_string] = self.create_client(connection_string)
        logging.debug('get_client #clients: %s' % (len(self.clients[thread_id]),))
        return self.clients[thread_id][connection_string]

    def make(self, **kwargs):
        started = dt.datetime.now()
        result = None
        try:
            result = self.makeimpl(**kwargs)
            if result is not None:
                logging.info(
                    '%s retrieved in %.0f ms' % (self.name, (dt.datetime.now() - started).total_seconds() * 1000))
            else:
                logging.info('Cannot retrieve  %s ' % self.name)
        except:
            logging.exception('Cannot retrieve sample %s %s' % (self.name, kwargs))
        return result

    def makeimpl(self, **kwargs):
        """
        Override and return a dictionary with the raw data retrieved from the data source, or None if there is no
        possibility to retrieve.
        """
        raise NotImplementedError()


class SQLDataSource(ThreadSafeDataSource):
    DRIVER = 'DRIVER={FreeTDS};ClientCharset=UTF-8;TDS_Version=7.0;'

    def __init__(self, connection_string):
        ThreadSafeDataSource.__init__(self)
        self.connection_string = connection_string
        self.create_client = lambda connection_string: pyodbc.connect(connection_string, getdata_str_size=50000)
        self.close_client = lambda connection: connection.close()

    def _execute(self, query, params, action):
        error_count = 0
        full_connection_string = self.DRIVER + self.connection_string
        while True:
            try:
                logging.debug('%s %s %s' % (self.__class__.__name__, query, params))
                connection = self.get_client(full_connection_string)
                with connection.cursor() as cursor:
                    cursor.execute(query, *params)
                    return action(cursor)
            except Exception as ex:
                if 'Write to the server failed' in str(ex) or 'Read from server failed' in str(
                        ex) or 'Communication link failure' in str(ex):
                    error_count += 1
                    if error_count == 1:
                        logging.info('Write to server failed, repeating')
                    else:
                        if error_count <= 5:
                            logging.info('Write to server failed, reopening connections and repeating')
                            self.delete_client(full_connection_string)
                        else:
                            raise
                else:
                    raise

    def get_row_as_dict(self, query, *params):
        return self._execute(query, params, lambda cursor: self._convert_row_to_dict(cursor.fetchone()))

    def get_rows_as_dict(self, query, *params):
        return self._execute(query, params, lambda cursor: self._convert_rows_to_dict(cursor.fetchall()))

    def _convert_row_to_dict(self, row):
        if row is not None:
            data = {}
            for d in row.cursor_description:
                logging.debug('%s\t%s' % (d[0], row.__getattribute__(d[0]),))
                if row.__getattribute__(d[0]) is not None:
                    data[d[0]] = row.__getattribute__(d[0])
            return data
        return None

    def _convert_rows_to_dict(self, rows):
        return [self._convert_row_to_dict(row) for row in rows if row is not None]

    def commit(self, ):
        try:
            self.get_client(self.DRIVER + self.connection_string).commit()
        except Exception as e:
            logging.exception(e)


class CouchBaseDataSource(ThreadSafeDataSource):

    def __init__(self, connection_string):
        ThreadSafeDataSource.__init__(self)
        self.connection_string = connection_string
        self.create_client = lambda connection_string: Bucket(self.connection_string, timeout=60)

    def get_document(self, key):
        bucket = self.get_client(self.connection_string)
        return bucket.get(key, quiet=True).value

    def get_multi(self, keys):
        bucket = self.get_client(self.connection_string)
        result = dict()
        for key, value in bucket.get_multi(keys, quiet=True).items():
            if value is not None:
                result[key] = value
        return result

    def query_view(self, design, view, key, include_docs=False):
        bucket = self.get_client(self.connection_string)
        return bucket.view_query(design, view, key=key, stale='ok', full_set=True,
            include_docs=include_docs)


class MongoDBDataSource(ThreadSafeDataSource):
    def __init__(self, connection_string, database, collection, socketTimeoutMS=20000):
        ThreadSafeDataSource.__init__(self)
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self.create_client = lambda connection_string: MongoClient(host=connection_string, document_class=dict,
            socketTimeoutMS=socketTimeoutMS)

    def find_one(self, filter=None, *args, **kwargs):
        result = self.get_client(self.connection_string)[self.database][self.collection].find_one(filter, *args,
            **kwargs)
        logging.debug(result)
        return result


class ClickhouseDataSource(ThreadSafeDataSource):
    def __init__(self, connection_string, timeout=5):
        ThreadSafeDataSource.__init__(self)
        self.connection_string = connection_string
        self.create_client = lambda connection_string: Connection(connection_string.split(':')[0],
            connection_string.split(':')[1],
            timeout=timeout)
        self.close_client = lambda connection: connection.close()

    def _select(self, query, params, action):
        logging.debug('%s %s %s' % (self.__class__.__name__, query, params))
        connection = self.get_client(self.connection_string)
        cur = connection.cursor()
        cur.select(query, *params)
        return action(cur)

    def get_row_as_dict(self, query, *params):
        return self._select(query, params, lambda cursor: cursor.fetchone())

    def get_rows_as_dict(self, query, *params):
        return self._select(query, params, lambda cursor: cursor.fetchall())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
