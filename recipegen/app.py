from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, Settings
from .models import RecommendRequest
from .pipeline import Recommender
from .system_api_models import GenerateRequest, SearchRequest


def create_app(settings: Settings | None = None, recommender: Recommender | None = None,
               *, catalog=None, generator=None, retriever=None, api_answerer=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        current = getattr(app.state, "recommender", None)
        if current is not None and hasattr(current.store, "close"):
            current.store.close()
        current_catalog = getattr(app.state, "catalog", None)
        if current_catalog is not None and hasattr(current_catalog, "close"):
            current_catalog.close()

    app = FastAPI(title="RecipeGen：基于知识图谱的食谱检索与生成系统", version="0.3.0", lifespan=lifespan)
    app.state.recommender = recommender
    app.state.catalog = catalog
    app.state.generator = generator
    app.state.retriever = retriever
    app.state.api_answerer = api_answerer

    def real_catalog():
        if app.state.catalog is None:
            from .catalog import RecipeCatalog
            app.state.catalog = RecipeCatalog(settings=settings)
        return app.state.catalog

    def real_generator():
        if app.state.generator is None:
            from .system_generation import RecipeGenerator
            app.state.generator = RecipeGenerator(project_root=PROJECT_ROOT)
        return app.state.generator

    def real_retriever():
        if app.state.retriever is None:
            from .graphrag_retrieval import RecipeGraphRAGRetriever
            from .text_embeddings import TextEmbedder
            app.state.retriever = RecipeGraphRAGRetriever(real_catalog(), embed_query=TextEmbedder().embed_query)
        return app.state.retriever

    def external_answerer():
        if app.state.api_answerer is None:
            from .graphrag_answer import GraphRAGAnswerer
            app.state.api_answerer = GraphRAGAnswerer(PROJECT_ROOT, settings=settings)
        return app.state.api_answerer

    def graph_unavailable():
        return HTTPException(503, "真实 Neo4j 食谱图谱暂不可用，请检查数据库连接；系统不会切换为人工示例数据。")

    @app.get("/api/system/status")
    def system_status():
        try:
            result = real_catalog().status()
            result["generation"] = real_generator().status()
            api_status = external_answerer().status()
            result["generation"]["local_model"] = result["generation"].get("model")
            result["generation"]["api_model"] = api_status.get("model")
            result["generation"].update({key: value for key, value in api_status.items() if key != "model"})
            from .text_embeddings import TextEmbedder
            embedding = TextEmbedder().status()
            retrieval = real_retriever().status()
            result["retrieval"] = {**retrieval, "default_mode": settings.retrieval_mode,
                                   "keyword_available": retrieval.get("keyword", {}).get("available", False),
                                   "semantic_available": retrieval.get("semantic", {}).get("available", False) and embedding["available"],
                                   "embedding_model": embedding}
            progress = PROJECT_ROOT / "reports" / "multimodal-progress.json"
            if progress.is_file():
                import json
                snapshot = json.loads(progress.read_text())
                result["media_processing"] = {key: snapshot.get(key) for key in (
                    "status", "scope", "processing_modalities", "completed_image_scope", "completed_full_scope")}
            return result
        except Exception:
            raise graph_unavailable() from None

    @app.post("/api/search")
    def search_recipes(request: SearchRequest):
        from .catalog import CatalogError
        from .graphrag_retrieval import GraphRAGUnavailable
        import time
        try:
            started = time.perf_counter()
            result = real_retriever().search(request.query, ingredients=request.ingredients,
                                             excluded_ingredients=request.excluded_ingredients, limit=request.limit,
                                             mode=request.retrieval_mode or settings.retrieval_mode)
            result["trace"] = [{"stage": "graphrag_retrieval", "status": result.get("retrieval", {}).get("status", "ok"),
                                "elapsed_ms": round((time.perf_counter()-started)*1000, 2)}]
            return result
        except GraphRAGUnavailable:
            raise HTTPException(503, "GraphRAG 语义检索尚未就绪，请检查文本模型和索引；也可选择关键词检索。") from None
        except CatalogError:
            raise graph_unavailable() from None
        except ValueError:
            raise HTTPException(422, "检索条件未通过校验") from None
        except Exception:
            raise graph_unavailable() from None

    @app.get("/api/recipes/{recipe_id}")
    def recipe_detail(recipe_id: str):
        from .catalog import CatalogError, RecipeNotFound
        try:
            return real_catalog().recipe(recipe_id)
        except RecipeNotFound:
            raise HTTPException(404, "图谱中没有这条食谱记录") from None
        except CatalogError:
            raise graph_unavailable() from None
        except ValueError:
            raise HTTPException(422, "菜谱 ID 无效") from None
        except Exception:
            raise graph_unavailable() from None

    @app.get("/api/system/graph")
    def system_graph(recipe_id: str | None = None):
        from .catalog import CatalogError, RecipeNotFound
        try:
            return real_catalog().graph(recipe_id)
        except RecipeNotFound:
            raise HTTPException(404, "图谱中没有这条食谱记录") from None
        except CatalogError:
            raise graph_unavailable() from None
        except ValueError:
            raise HTTPException(422, "菜谱 ID 无效") from None
        except Exception:
            raise graph_unavailable() from None

    @app.post("/api/generate")
    def generate_recipe(request: GenerateRequest):
        from .catalog import CatalogError, RecipeNotFound, normalize_ingredient, parse_query_filters
        from .graphrag_retrieval import GraphRAGUnavailable
        import time
        started = time.perf_counter()
        try:
            if request.recipe_ids:
                recipes = [real_catalog().recipe(identifier) for identifier in request.recipe_ids]
                implicit_excluded = parse_query_filters(request.question)["excluded"]
                excluded = {normalize_ingredient(name) for name in request.excluded_ingredients} | set(implicit_excluded)
                if excluded:
                    recipes = [recipe for recipe in recipes if not excluded.intersection(
                        normalize_ingredient(mention.get("name", mention.get("normalized_name", "")))
                        for mention in recipe.get("ingredient_mentions", []))]
                retrieval = {"selection": "explicit_recipe_ids", "excluded_ingredients": sorted(excluded),
                             "inventory_or_exclusion_verified": False, "mode": request.retrieval_mode or settings.retrieval_mode,
                             "effective_mode": "selected_recipes", "status": "ok", "method": "verified_recipe_ids"}
            else:
                retrieved = real_retriever().search(request.question, ingredients=request.ingredients,
                                                    excluded_ingredients=request.excluded_ingredients,
                                                    limit=max(6, request.limit), mode=request.retrieval_mode or settings.retrieval_mode)
                recipes = retrieved["recipes"]
                retrieval = {**retrieved.get("retrieval", {}), **retrieved.get("applied_filters", {})}
            retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
            kwargs = {"ingredients": request.ingredients,
                      "excluded_ingredients": retrieval.get("excluded_ingredients", request.excluded_ingredients),
                      "limit": request.limit}
            if request.mode == "api":
                result = external_answerer().answer(request.question, recipes, retrieval_context=retrieval, **kwargs)
            else:
                result = real_generator().generate(request.question, recipes, mode=request.mode, **kwargs)
            result["retrieval"] = retrieval
            result.setdefault("trace", []).insert(0, {"stage": "graph_retrieval", "status": "ok",
                                                        "elapsed_ms": retrieval_ms})
            return result
        except RecipeNotFound:
            raise HTTPException(404, "所选食谱已不存在，请重新检索") from None
        except GraphRAGUnavailable:
            raise HTTPException(503, "GraphRAG 语义检索尚未就绪，请检查文本模型和索引；也可选择关键词检索。") from None
        except CatalogError:
            raise graph_unavailable() from None
        except ValueError:
            raise HTTPException(422, "食谱证据或生成条件未通过校验") from None
        except Exception:
            raise graph_unavailable() from None

    def service() -> Recommender:
        if app.state.recommender is None:
            app.state.recommender = Recommender(settings)
        return app.state.recommender

    def unavailable() -> HTTPException:
        return HTTPException(status_code=503, detail="图谱读取失败。请检查所选后端、数据导入及数据库配置；本服务不会自动切换到示例图谱。")

    @app.get("/api/status")
    def status():
        try:
            return service().status()
        except Exception:
            raise unavailable() from None

    @app.post("/api/recommend")
    def recommend(request: RecommendRequest):
        try:
            return service().recommend(request)
        except ValueError:
            raise HTTPException(422, "输入条件或图谱记录未通过校验") from None
        except Exception:
            raise unavailable() from None

    @app.get("/api/graph")
    def graph(recipe_id: str | None = None):
        try:
            return service().store.graph(recipe_id)
        except Exception:
            raise unavailable() from None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    static = PROJECT_ROOT / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(static / "index.html")

    return app
