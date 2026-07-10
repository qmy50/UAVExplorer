#include <iostream>
#include <vector>

using namespace std;

struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution{
public: 
    vector<vector<int>> levelOrder(TreeNode* root){
        int MAXNUM = 100;
        TreeNode *queue[MAXNUM];
        int left = 0, right = 0;
        TreeNode *tmp = root;
        queue[0] = root;
        right++;
        vector<vector<int>> result;
        int size = 0;
        while(left < right){
            //std::cout << "Hi" << std::endl;
            vector<int> tmp_list;
            size = right - left;
            for (int i = 0; i < size;i++)
            {
                tmp = queue[left++];
                tmp_list.push_back(tmp->val);
                //std::cout << "The val is " << tmp->val << std::endl;
                if(tmp->left != nullptr){
                    queue[right++] = tmp->left;
                    //std::cout << "The left child val is" << tmp->left->val << std::endl;
                }
                if(tmp->right != nullptr){
                    queue[right++] = tmp->right;
                    //std::cout << "The right child val is" << tmp->right->val << std::endl;
                }
            }
            result.push_back(tmp_list);
            // std::cout << "The result size is " << result.size() << endl;
        }
        return result;
    }   
};

int main(int argc,char* argvp[]){
    TreeNode* root(new TreeNode(3));
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);

    vector<vector<int>> result = {};
    Solution solution;
    std::cout << "Hello" << std::endl;
    std::cout << "The value is " << root->left->val << std::endl;
    result = solution.levelOrder(root);
    for(const auto item:result){
        for(const int num:item){
            std::cout << num << ',';
        }
        std::cout << "Change" << std::endl;
        std::cout << std::endl;
    }
    return 0;
}