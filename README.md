# Template repository for Python projects
Use this repo to create new python projects with ready-to-go build and deploy pipelines, configured to use Poetry as a dependency manager.

## Checklist

Remember to do the following after creating a new repo:

- :heavy_check_mark: Rename project folder `python_project` to `**name_of_project**`
- :heavy_check_mark: Search and replace `<python_project>` and `python_project` with your desired package name (repository name)
- :heavy_check_mark: Manually create a github release targeting the very first commit in your repo, setting tag to `v0.0.0`
- :heavy_check_mark: Select build and deploy pipelines to use in `workflows/` dir. Usually you will need `build.yaml` and `deploy-azure-artifacts.yaml`. 
  - Remove dummy code and uncomment real code
  - Remove pipelines you won't need
- :heavy_check_mark: Update this README.

Happy coding!
