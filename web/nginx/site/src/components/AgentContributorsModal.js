import React, {Component} from "react";
import {BASE_URL, getJWT, refreshJWT} from "../Util";
import {toast} from "react-semantic-toasts";
import {Button, Grid, Header, Icon, List, Modal, Search} from "semantic-ui-react";

class ContributorList extends Component {
  // props:
  // contributors: List[User]
  // onDelete(contributor)

  render = () => {
    if (this.props.contributors.length === 0) {
      return ('This agent currently has no contributors');
    }

    let contributors = [];
    this.props.contributors.forEach(contributor => {
      if (!('user' in contributor)) {
        return;
      }

      let user = contributor.user;
      contributors.push((
        <List.Item key={user.id}>
          <Button basic negative icon='cancel' onClick={() => this.props.onDelete(contributor)} size='mini'/>
          {' '}
          {user.username + ' (' + user.first_name + ' ' + user.last_name + ')'}
        </List.Item>
      ));
    });

    return (
      <List>
        {contributors}
      </List>
    );
  };
}

export default class AgentContributorsModal extends Component {
  constructor(props) {
    // props:
    // agent
    super(props);
    this.state = this.initialState();
  }

  initialState = () => {
    return {
      modalOpen: false,
      contributors: [],
      name: '',
      description: '',
      usersSearchResult: [],
      isLoading: false,
    };
  };

  resetState = () => {
    this.setState(this.initialState);
  };

  componentDidMount = () => {
    fetch(BASE_URL + '/api/contributors/', {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        console.error("Unable to fetch contributors: ", response);
        return;
      }

      response.json().then(contributors => {
        this.setState({contributors: contributors.results});
      })
    });
  };

  usersSearch = (search) => {
    this.setState({isLoading: true});
    fetch(BASE_URL + '/api/users/?page_size=5&search=' + search, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
    }).then(response => {
      this.setState({isLoading: false});
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        console.error("Unable to fetch users: ", response);
        return;
      }

      response.json().then(users => {
        let searchResults = [];
        users.results.forEach(user => {
          searchResults.push({
            title: user.username,
            description: user.first_name + ' ' + user.last_name + ' - ' + user.email,
            user: user,
          });
        });
        this.setState({usersSearchResult: searchResults});
      })
    });
  };

  addContributor = (contributor) => {
    fetch(BASE_URL + '/api/contributors/', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
      body: JSON.stringify({
        agent_id: this.props.agent.id,
        user_id: contributor.user.id,
      }),
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 201) {
        console.error("Unable to add contributor: ", response);
        return;
      }

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Successfully added contributor</p>
        },
      );

      response.json().then(body => {
        this.setState({contributors: [...this.state.contributors, body]});
      });
    });
  };

  removeContributor = (contributor) => {
    fetch(BASE_URL + '/api/contributors/' + contributor.id + '/', {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
      body: JSON.stringify({
        agent_id: this.props.agent.id,
        user_id: contributor.id,
      }),
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 204) {
        console.error("Unable to delete contributor: ", response);
        return;
      }

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Successfully deleted contributor</p>
        },
      );

      let contributors = [...this.state.contributors];
      let index = contributors.indexOf(contributor);
      if (index !== -1) {
        contributors.splice(index, 1);
        this.setState({contributors: contributors});
      }
    });
  };

  handleUserResultSelect = (event, {result}) => {
    this.addContributor(result);
  };

  handleUserSearchChange = (event) => {
    this.usersSearch(event.target.value);
  };

  render = () => {
    return (
      <Modal open={this.state.modalOpen}
             trigger={
               <Button fluid style={{marginTop: '5px'}} onClick={() =>
                 this.setState({modalOpen: true})
               }>
                 <Icon name='group'/>{' '}Contributors
               </Button>
             }
             size='small'>
        <Header icon='group' content='Contributors'/>
        <Modal.Content>
          <Grid columns={2}>
            <Grid.Column>
              <p>Search for contributors to add to this project.</p>
              <Search
                loading={this.state.isLoading}
                onResultSelect={this.handleUserResultSelect}
                onSearchChange={this.handleUserSearchChange}
                results={this.state.usersSearchResult}
              />
            </Grid.Column>
            <Grid.Column>
              <ContributorList
                contributors={this.state.contributors}
                onDelete={contributor => this.removeContributor(contributor)}
              />
            </Grid.Column>
          </Grid>
        </Modal.Content>
        <Modal.Actions>
          <Button color='green' inverted onClick={() => this.resetState()}>
            <Icon name='check'/>{' '}Done
          </Button>
        </Modal.Actions>
      </Modal>
    );
  };
}
