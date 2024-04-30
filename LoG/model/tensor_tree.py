import torch
import torch.nn as nn

class TensorTree(nn.Module):
    def __init__(self, max_child=2, max_level=20):
        super().__init__()
        num_points = 0
        self.max_child = max_child
        self.max_level = max_level
        root_index = torch.arange(num_points, dtype=torch.int32)
        self.register_buffer('root_index', root_index)
        for name in ['node_index', 'index_parent']:
            index = torch.zeros((num_points,), dtype=torch.int32) - 1
            self.register_buffer(name, index)
        for name in ['local_index', 'depth']:
            index = torch.zeros((num_points,), dtype=torch.int8)
            self.register_buffer(name, index)
        self.keys = ['node_index', 'index_parent', 'local_index', 'depth']
        tree = torch.zeros((0, max_child), dtype=torch.int32) - 1
        self.register_buffer('tree', tree)
        self.min_resolution_pixel = 3
        self.log_query = False

    @property
    def num_points(self):
        return self.node_index.shape[0]
    
    @property
    def num_nodes(self):
        return self.tree.shape[0]
    
    def initialize(self, data, flag=None):
        root_index = torch.arange(data.shape[0], dtype=torch.int32, device=self.root_index.device)
        if flag is None:
            print(f'[{self.__class__.__name__}] initialize the tree with {data.shape[0]} points...')
        else:
            print(f'[{self.__class__.__name__}] initialize the tree with {flag.sum()}/{data.shape[0]} points...')
            root_index = root_index[flag]
        self.root_index.set_(root_index)
        for key in self.keys:
            index = torch.zeros((data.shape[0],), dtype=getattr(self, key).dtype, device=getattr(self, key).device) - 1
            getattr(self, key).set_(index)
        self.depth.fill_(0)

    def __repr__(self):
        num_parents = (self.node_index > -1).sum()
        num_leaves = (self.node_index == -1).sum()
        ret = f'Tree: {self.node_index.shape[0]} points:{num_parents} parents, {num_leaves} leaves, {self.num_nodes} nodes'
        return ret
    
    def print_level(self):
        depth_max = self.depth.max().item()
        print(f'[{self.__class__.__name__}] tree level: {depth_max+1}')
        for i in range(depth_max+1):
            print('  ' * (i+1), f'level {i}: {self.node_index[self.depth == i].shape[0]}')

    @property
    def is_leaf(self):
        return self.node_index == -1
    
    @property
    def is_root(self):
        return self.index_parent == -1

    def split(self, parent_index):
        device = self.tree.device
        num_split = len(parent_index)
        if isinstance(parent_index, list):
            parent_index = torch.tensor(parent_index, dtype=torch.long, device=device)
        # update the node index
        node_index = torch.arange(0, num_split, device=device, dtype=torch.int32) + self.num_nodes
        self.node_index[parent_index] = node_index
        # update the node table
        child_index = torch.arange(0, num_split * self.max_child,
                            device=device, dtype=torch.int32) + self.num_points
        child_index = child_index.reshape(num_split, self.max_child)
        tree_new = torch.cat([self.tree, child_index])
        self.tree.set_(tree_new)
        # update the data
        # add node table
        num_new = num_split * self.max_child
        index_parent = parent_index[:, None].repeat(1, self.max_child).reshape(-1).int()
        depth = self.depth[parent_index][:, None].repeat(1, self.max_child).reshape(-1) + 1
        local_index = torch.arange(0, self.max_child, device=device, dtype=torch.int8)[None]\
                                .repeat(num_split, 1).reshape(-1)
        node_index = torch.zeros((num_new,), dtype=torch.int32, device=device) - 1
        self.node_index.set_(torch.cat([self.node_index, node_index]))
        self.index_parent.set_(torch.cat([self.index_parent, index_parent]))
        self.depth.set_(torch.cat([self.depth, depth]))
        self.local_index.set_(torch.cat([self.local_index, local_index]))

    def remove(self, index):
        parent_index = self.index_parent[index].long()
        local_index = self.local_index[index].long()
        node_index = self.node_index[parent_index].long()
        # reset the node of parent
        # 
        children_index = self.tree[node_index, local_index].long()
        # reset the tree
        self.tree[node_index, local_index] = -1
        flag_keep = torch.ones_like(self.node_index, dtype=torch.bool)
        flag_keep[children_index] = False
        for key in self.keys:
            getattr(self, key).set_(getattr(self, key)[flag_keep])
        # update the tree
        left_index = torch.cumsum(flag_keep, 0) - 1
        left_index[self.tree.long()]
        # record the node
        flag_node_keep = self.tree > -1
        self.tree[flag_node_keep] = left_index[self.tree[flag_node_keep].long()].int()
        # update the index of parent
        flag_nonroot = self.index_parent > -1
        self.index_parent[flag_nonroot] = left_index[self.index_parent[flag_nonroot].long()].int()
        # update tree and node
        flag_remove_tree = self.node_index != -1
        flag_remove_tree_ = (self.tree[self.node_index[flag_remove_tree].long()] < 0).all(dim=-1)
        flag_remove_tree[flag_remove_tree.clone()] = flag_remove_tree_
        self.node_index[flag_remove_tree] = -1
    
    def split_and_remove(self, flag_split, flag_remove):
        flag_remove = flag_remove & self.is_leaf & (~self.is_root)
        flag_split = flag_split & self.is_leaf & (self.depth < self.max_level)
        index_split = torch.where(flag_split)[0]
        index_remove = torch.where(flag_remove)[0]
        print(f' -> [{self.__class__.__name__}] split: {index_split.shape[0]} remove: {index_remove.shape[0]}')
        # ATTN: remove must after the split operation
        self.split(index_split)
        self.remove(index_remove)
        return flag_split, flag_remove

    @torch.no_grad()
    def _query_tree_torch(self, model, index, camera, min_resolution_pixel, max_depth):
        indices_list = []
        level = 1
        while True:
            if level > self.max_level or level > max_depth:
                # just use the current index
                indices_list.append(index)
                break
            index_node = self.node_index[index].long()
            id_child = self.tree[index_node].flatten().long()
            id_child = id_child[id_child != -1]
            scale3d, scale2d = model.compute_radius(id_child, camera, level=level)
            flag_is_small = scale2d < min_resolution_pixel
            flag_is_leaf = self.node_index[id_child] == -1
            flag_keep = flag_is_small | flag_is_leaf
            flag_go_to_next = ~(flag_keep)
            if self.log_query:
                print(f'level {level:2d}: query {index_node.shape[0]} -> {id_child.shape[0]:10d} nodes')
                print(f'        scale3d:[{scale3d.min().item():.4f}~{scale3d.mean().item():.4f}~{scale3d.max().item():.4f}]')
                print(f'        scale2d:[{scale2d.min().item():6.1f}~{scale2d.mean().item():6.1f}~{scale2d.max().item():6.1f}]')
                print(f'        {flag_is_small.sum().item():6d} nodes < {min_resolution_pixel}pixel; {flag_is_leaf.sum().item():6d} nodes are leaves')
                print(f'        keep {flag_keep.sum()}, next {flag_go_to_next.sum()}')
            indices_list.append(id_child[flag_keep])
            if flag_go_to_next.sum() == 0:
                break
            index = id_child[flag_go_to_next].long()
            level += 1
        if len(indices_list) == 0:
            return index
        else:
            indices_list = torch.cat(indices_list, dim=0)
        return indices_list

    def traverse(self, model, root_index, rasterizer, max_depth=1000):
        # compute_radius
        radius3d, radius2d = model.compute_radius(root_index, rasterizer, level=0)
        if self.log_query:
            print(f'level {0:2d}: query {root_index.shape[0]} root, radius: [{radius2d.min():.1f}~{radius2d.mean():.1f}~{radius2d.max():.1f}]')
        flag_root_has_child = self.node_index[root_index] != -1
        flag_root_no_child = ~flag_root_has_child
        flag_root_small = radius2d < self.min_resolution_pixel
        flag_go_to_next = (~flag_root_small) & flag_root_has_child
        flag_keep = ~flag_go_to_next
        index_root_keep = root_index[flag_keep]
        index_root_next = root_index[flag_go_to_next]
        if self.log_query:
            print(f'        skip {index_root_keep.shape[0]}, next {index_root_next.shape[0]}')
        # select the index
        index_child = self._query_tree_torch(model, index_root_next, rasterizer, 
                            min_resolution_pixel=self.min_resolution_pixel, max_depth=max_depth)
        index_concat = torch.cat([index_root_keep, index_child], dim=0)
        depth = self.depth[index_concat].float().mean()
        if self.log_query:
            print(f' query mean depth: {depth}')
        return index_concat