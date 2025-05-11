import os
import logging
from typing import Optional, List

import json
import yaml
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from franchise import GeminiFranchiseService, logger


class GeminiFewShotFranchiseService(GeminiFranchiseService):
    def __init__(self, api_key: str = None, config: dict = None):
        super().__init__(api_key, config)
        self.qa_vectorstore = self._load_qa_vectorstore()
        self.prompt_template = self._load_prompt_templates("./data/prompt_template.yaml")

    def _load_qa_vectorstore(self):
        try:
            logger.info("[QA] 벡터스토어 로딩")
            vs = Chroma(
                persist_directory="./vector_db/qa_knowledge_base",
                embedding_function=self.embeddings,
                collection_name="contracts_qa_collection"
            )
            count = vs._collection.count()
            logger.info(f"[QA] 문서 수: {count}")
            return vs if count > 0 else None
        except Exception as e:
            logger.error(f"[QA] 벡터스토어 로딩 실패: {str(e)}")
            return None

    def _load_prompt_templates(self, yaml_path: str) -> dict:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def retrieve_context(self, vectorstore: Chroma, query: str, filter: Optional[dict] = None) -> List[Document]:
        search_kwargs = {"query": query, "k": self.vectorstore_search_k}
        if filter:
            search_kwargs["filter"] = filter
        return vectorstore.similarity_search(**search_kwargs)

    def build_prompt_from_template(
        self,
        template_str: str,
        user_query: str,
        context: str,
        docs: Optional[List[Document]] = None,
        max_examples: int = 3
    ) -> str:
        examples_text = ""

        if docs:
            qa_examples = []
            for doc in docs:
                try:
                    qas = json.loads(doc.metadata.get("QAs", "[]"))
                    for qa in qas:
                        q = qa.get("QUESTION", "").strip()
                        a = qa.get("ANSWER", "").strip()
                        if q and a:
                            qa_examples.append(f"Q: {q}\nA: {a}")
                        if len(qa_examples) >= max_examples:
                            break
                except json.JSONDecodeError:
                    continue
                if len(qa_examples) >= max_examples:
                    break

            examples_text = "\n\n".join(qa_examples) if qa_examples else "(예시 없음)"

        return (
            template_str
            .replace("%examples%", examples_text)
            .replace("%context%", context.strip())
            .replace("%question%", user_query.strip())
        )

    def answer_question_with_prompt(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)

        return response.text

    def inference(self, query: str) -> str:
        context_docs = self.retrieve_context(self.chroma_vectorstore, query)
        if not context_docs:
            return "❌ 관련 문서를 찾지 못했습니다."

        top_doc = context_docs[0]
        context = top_doc.page_content
        attrb_mnno = top_doc.metadata.get("ATTRB_MNNO")

        if self.qa_vectorstore:
            qa_docs = self.retrieve_context(self.qa_vectorstore, query, filter={"ATTRB_MNNO": attrb_mnno})
            if not qa_docs:
                logger.warning("⚠ 필터 조건에 맞는 QA 문서가 없어 전체 QA 벡터스토어에서 재검색합니다.")
                qa_docs = self.retrieve_context(self.qa_vectorstore, query)

            template = self.prompt_template["fewshot_template"]
            prompt = self.build_prompt_from_template(template, query, context, docs=qa_docs)
        else:
            template = self.prompt_template["basic_template"]
            prompt = self.build_prompt_from_template(template, query, context)

        print(f"======프롬프트======")
        print(f"{prompt}")

        ##응답 생성
        response = self.model.generate_content(prompt)

        output =  {
            "original_text": context,
            "question": query,
            "answer": response.text
        }

        return output

        # return self.answer_question_with_prompt(prompt)