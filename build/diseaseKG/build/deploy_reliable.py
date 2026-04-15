# coding: utf-8

import os
import json
from neo4j import GraphDatabase
import time
import sys

class MedicalGraphReliable:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/medical.json')
        self.uri = "neo4j+s://627658bb.databases.neo4j.io"
        self.user = "627658bb"
        self.password = "35No69NaLaoasxQqW-JhcjbxgQjeY_WzUVGHYtKWeNo"
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.batch_size = 500

    def close(self):
        if self.driver:
            self.driver.close()

    def log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def clear_database(self):
        self.log("="*60)
        self.log("清空数据库以保证完整性...")
        self.log("="*60)
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            before = result.single()['count']
            self.log(f"清空前列数: {before}")
            
            session.run("MATCH (n) DETACH DELETE n")
            
            result = session.run("MATCH (n) RETURN count(n) as count")
            after = result.single()['count']
            self.log(f"清空后列数: {after}")
        self.log("数据库已清空")

    def read_nodes(self):
        self.log("开始读取数据文件...")
        drugs = []
        foods = []
        checks = []
        departments = []
        producers = []
        diseases = []
        symptoms = []
        cures = []
        disease_infos = []
        
        rels = {
            'department': [],
            'noteat': [],
            'doeat': [],
            'recommandeat': [],
            'commonddrug': [],
            'recommanddrug': [],
            'check': [],
            'drug_producer': [],
            'cureway': [],
            'symptom': [],
            'acompany': [],
            'category': []
        }

        count = 0
        for data in open(self.data_path, 'rb'):
            disease_dict = {}
            count += 1
            data_json = json.loads(data)
            disease = data_json['name']
            disease_dict['name'] = disease
            diseases.append(disease)
            disease_dict['desc'] = data_json.get('desc', '')
            disease_dict['prevent'] = data_json.get('prevent', '')
            disease_dict['cause'] = data_json.get('cause', '')
            disease_dict['easy_get'] = data_json.get('easy_get', '')
            disease_dict['cure_lasttime'] = data_json.get('cure_lasttime', '')
            disease_dict['cured_prob'] = data_json.get('cured_prob', '')

            if 'symptom' in data_json:
                symptoms += data_json['symptom']
                for symptom in data_json['symptom']:
                    rels['symptom'].append([disease, symptom])

            if 'acompany' in data_json:
                for acompany in data_json['acompany']:
                    rels['acompany'].append([disease, acompany])

            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']
                if len(cure_department) == 1:
                    rels['category'].append([disease, cure_department[0]])
                if len(cure_department) == 2:
                    big = cure_department[0]
                    small = cure_department[1]
                    rels['department'].append([small, big])
                    rels['category'].append([disease, small])
                departments += cure_department

            if 'cure_way' in data_json:
                cure_way = data_json['cure_way']
                cures += cure_way
                for cure in cure_way:
                    rels['cureway'].append([disease, cure])

            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']
                for drug in common_drug:
                    rels['commonddrug'].append([disease, drug])
                drugs += common_drug

            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                drugs += recommand_drug
                for drug in recommand_drug:
                    rels['recommanddrug'].append([disease, drug])

            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']
                for _not in not_eat:
                    rels['noteat'].append([disease, _not])
                foods += not_eat

                do_eat = data_json['do_eat']
                for _do in do_eat:
                    rels['doeat'].append([disease, _do])
                foods += do_eat

                recommand_eat = data_json['recommand_eat']
                for _recommand in recommand_eat:
                    rels['recommandeat'].append([disease, _recommand])
                foods += recommand_eat

            if 'check' in data_json:
                check = data_json['check']
                for _check in check:
                    rels['check'].append([disease, _check])
                checks += check

            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']
                producer = [i.split('(')[0] for i in drug_detail]
                rels['drug_producer'] += [[i.split('(')[0], i.split('(')[-1].replace(')', '')] for i in drug_detail]
                producers += producer

            disease_infos.append(disease_dict)

        self.log(f"数据读取完成，共 {count} 条疾病记录")
        return (
            set(drugs), set(foods), set(checks), set(departments), 
            set(producers), set(symptoms), set(diseases), set(cures),
            disease_infos, rels
        )

    def create_diseases_nodes(self, disease_infos):
        count = 0
        total = len(disease_infos)
        self.log(f"开始创建 Disease 节点，共 {total} 个")
        with self.driver.session() as session:
            for i in range(0, total, self.batch_size):
                batch = disease_infos[i:i + self.batch_size]
                session.run(
                    "UNWIND $batch AS disease "
                    "MERGE (n:Disease {name: disease.name}) "
                    "SET n.desc = disease.desc, n.prevent = disease.prevent, n.cause = disease.cause, "
                    "n.easy_get = disease.easy_get, n.cure_lasttime = disease.cure_lasttime, "
                    "n.cured_prob = disease.cured_prob",
                    batch=batch
                )
                count += len(batch)
                self.log(f"  Disease 节点: {count}/{total}")
        self.log(f"Disease 节点创建完成: {count}/{total}")

    def create_simple_nodes(self, label, nodes):
        count = 0
        total = len(nodes)
        self.log(f"开始创建 {label} 节点，共 {total} 个")
        node_list = [{'name': n} for n in nodes]
        with self.driver.session() as session:
            for i in range(0, total, self.batch_size):
                batch = node_list[i:i + self.batch_size]
                session.run(
                    "UNWIND $batch AS item "
                    "MERGE (n:" + label + " {name: item.name})",
                    batch=batch
                )
                count += len(batch)
                self.log(f"  {label} 节点: {count}/{total}")
        self.log(f"{label} 节点创建完成: {count}/{total}")

    def create_graphnodes(self):
        self.log("="*60)
        self.log("开始创建节点")
        self.log("="*60)
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, Cures, disease_infos, rels = self.read_nodes()
        self.create_diseases_nodes(disease_infos)
        self.create_simple_nodes('Drug', Drugs)
        self.create_simple_nodes('Food', Foods)
        self.create_simple_nodes('Check', Checks)
        self.create_simple_nodes('Department', Departments)
        self.create_simple_nodes('Producer', Producers)
        self.create_simple_nodes('Symptom', Symptoms)
        self.create_simple_nodes('Cure', Cures)
        return rels

    def create_relationships(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        unique_edges = list(set(set_edges))
        total = len(unique_edges)
        self.log(f"开始创建 {rel_type} 关系，共 {total} 条")
        
        edge_list = []
        for edge in unique_edges:
            p, q = edge.split('###')
            edge_list.append({'p': p, 'q': q})
        
        with self.driver.session() as session:
            for i in range(0, total, self.batch_size):
                batch = edge_list[i:i + self.batch_size]
                session.run(
                    "UNWIND $batch AS edge "
                    "MATCH (p:" + start_node + " {name: edge.p}), (q:" + end_node + " {name: edge.q}) "
                    "MERGE (p)-[r:" + rel_type + " {name: $rel_name}]->(q)",
                    batch=batch, rel_name=rel_name
                )
                count += len(batch)
                self.log(f"  {rel_type} 关系: {count}/{total}")
        self.log(f"{rel_type} 关系创建完成: {count}/{total}")

    def create_all_relationships(self, rels):
        self.log("="*60)
        self.log("开始创建关系")
        self.log("="*60)
        self.create_relationships('Disease', 'Food', rels['recommandeat'], 'recommand_eat', '推荐食谱')
        self.create_relationships('Disease', 'Food', rels['noteat'], 'no_eat', '忌吃')
        self.create_relationships('Disease', 'Food', rels['doeat'], 'do_eat', '宜吃')
        self.create_relationships('Department', 'Department', rels['department'], 'belongs_to', '属于')
        self.create_relationships('Disease', 'Drug', rels['commonddrug'], 'common_drug', '常用药品')
        self.create_relationships('Producer', 'Drug', rels['drug_producer'], 'drugs_of', '生产药品')
        self.create_relationships('Disease', 'Drug', rels['recommanddrug'], 'recommand_drug', '好评药品')
        self.create_relationships('Disease', 'Check', rels['check'], 'need_check', '诊断检查')
        self.create_relationships('Disease', 'Symptom', rels['symptom'], 'has_symptom', '症状')
        self.create_relationships('Disease', 'Disease', rels['acompany'], 'acompany_with', '并发症')
        self.create_relationships('Disease', 'Department', rels['category'], 'belongs_to', '所属科室')
        self.create_relationships('Disease', 'Cure', rels['cureway'], 'cure_way','治疗方法')

    def verify_deployment(self):
        self.log("="*60)
        self.log("验证部署完整性...")
        self.log("="*60)
        with self.driver.session() as session:
            result = session.run('MATCH (n) RETURN count(n) as node_count')
            node_count = result.single()['node_count']
            self.log(f"节点总数: {node_count}")
            
            result = session.run('MATCH (n:Disease) RETURN count(n) as count')
            self.log(f"  Disease: {result.single()['count']}")
            
            result = session.run('MATCH (n:Drug) RETURN count(n) as count')
            self.log(f"  Drug: {result.single()['count']}")
            
            result = session.run('MATCH (n:Food) RETURN count(n) as count')
            self.log(f"  Food: {result.single()['count']}")
            
            result = session.run('MATCH (n:Check) RETURN count(n) as count')
            self.log(f"  Check: {result.single()['count']}")
            
            result = session.run('MATCH (n:Department) RETURN count(n) as count')
            self.log(f"  Department: {result.single()['count']}")
            
            result = session.run('MATCH (n:Producer) RETURN count(n) as count')
            self.log(f"  Producer: {result.single()['count']}")
            
            result = session.run('MATCH (n:Symptom) RETURN count(n) as count')
            self.log(f"  Symptom: {result.single()['count']}")
            
            result = session.run('MATCH (n:Cure) RETURN count(n) as count')
            self.log(f"  Cure: {result.single()['count']}")
            
            result = session.run('MATCH ()-[r]->() RETURN count(r) as rel_count')
            rel_count = result.single()['rel_count']
            self.log(f"关系总数: {rel_count}")
            
            return node_count, rel_count

if __name__ == '__main__':
    start_time = time.time()
    print("="*60)
    print("DiseaseKG 医疗知识图谱可靠部署")
    print("="*60)
    
    handler = MedicalGraphReliable()
    try:
        handler.clear_database()
        rels = handler.create_graphnodes()
        handler.create_all_relationships(rels)
        handler.verify_deployment()
        
        handler.log("="*60)
        handler.log("部署完成！")
        handler.log(f"总耗时: {time.time() - start_time:.2f} 秒")
        handler.log("="*60)
    except Exception as e:
        handler.log(f"部署失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        handler.close()
